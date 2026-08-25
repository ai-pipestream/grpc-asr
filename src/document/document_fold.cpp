#include "document/document_fold.h"

#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace asr::doc {

namespace asrv1 = ai::pipestream::asr::v1;
namespace docv1 = ai::pipestream::document::v1;

namespace {

constexpr char kSchemaName[] = "docling_document_v2";
constexpr char kCollector[] = "asr";
// Names the engine signal CollectorSource.raw_score carries, so a consumer
// reading raw_score knows it is a mean token log probability and never
// mistakes it for a probability.
constexpr char kRawScoreKind[] = "avg_logprob";

std::string trimmed(const std::string& text) {
    const char* whitespace = " \t\r\n";
    size_t begin = text.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return "";
    }
    size_t end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1);
}

void set_number(google::protobuf::Map<std::string, google::protobuf::Value>* fields,
                const std::string& key, double value) {
    (*fields)[key].set_number_value(value);
}

void set_string(google::protobuf::Map<std::string, google::protobuf::Value>* fields,
                const std::string& key, const std::string& value) {
    (*fields)[key].set_string_value(value);
}

// Unicode code points in [from, to): every byte that is not a UTF-8
// continuation byte starts one. Charspans are code point offsets, not byte
// offsets, so a transcript with accents or CJK still highlights correctly.
int32_t code_points(const std::string& text, size_t from, size_t to) {
    int32_t count = 0;
    for (size_t i = from; i < to && i < text.size(); i++) {
        if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
            count++;
        }
    }
    return count;
}

// Fills a provenance entry's media span. Media has no pages, so page_no
// stays 0 and no bounding box is invented; the time range is the real
// locator. Zero-length spans are legal here (a keyframe is an instant) and
// are not widened.
void time_span(docv1::ProvenanceItem* prov, uint64_t start_ms, uint64_t end_ms,
               const std::string& speaker) {
    docv1::TimeSpan* span = prov->mutable_time();
    span->set_start_ms(static_cast<double>(start_ms));
    span->set_end_ms(static_cast<double>(end_ms));
    if (!speaker.empty()) {
        span->set_speaker(speaker);
    }
}

// Maps an ISO 639-1 code onto the schema's language enum by name; the enum
// value names are the uppercased codes. A code the enum does not know
// (whisper also emits three-letter codes such as "yue") leaves the enum
// unspecified, and code_raw keeps the string either way.
docv1::HumanLanguageLabel language_label(const std::string& code) {
    std::string primary = code.substr(0, code.find('-'));
    std::string name = "HUMAN_LANGUAGE_LABEL_";
    for (char c : primary) {
        name += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    docv1::HumanLanguageLabel label = docv1::HUMAN_LANGUAGE_LABEL_UNSPECIFIED;
    return docv1::HumanLanguageLabel_Parse(name, &label) ? label
                                                         : docv1::HUMAN_LANGUAGE_LABEL_UNSPECIFIED;
}

}  // namespace

AsrDocumentFold::AsrDocumentFold(std::string model, std::string version, FoldOptions options)
    : model_(std::move(model)), version_(std::move(version)), options_(options) {
    document_.set_schema_name(kSchemaName);
    docv1::GroupItem* body = document_.mutable_body();
    body->set_self_ref("#/body");
    body->set_content_layer(docv1::CONTENT_LAYER_BODY);
    docv1::GroupItem* furniture = document_.mutable_furniture();
    furniture->set_self_ref("#/furniture");
    furniture->set_content_layer(docv1::CONTENT_LAYER_FURNITURE);
}

void AsrDocumentFold::set_source(std::string filename, std::string mimetype,
                                 uint64_t binary_hash) {
    document_.set_name(filename);
    docv1::DocumentOrigin* origin = document_.mutable_origin();
    origin->set_filename(std::move(filename));
    origin->set_mimetype(std::move(mimetype));
    origin->set_binary_hash(binary_hash);
}

void AsrDocumentFold::consume(const asrv1::TranscribeResponse& event) {
    if (finished_) {
        return;
    }
    switch (event.event_case()) {
        case asrv1::TranscribeResponse::kMediaInfo:
            return on_media_info(event.media_info());
        case asrv1::TranscribeResponse::kFinalSegment:
            return on_final_segment(event.final_segment());
        case asrv1::TranscribeResponse::kKeyframe:
            return on_keyframe(event.keyframe());
        case asrv1::TranscribeResponse::kComplete:
            return on_complete(event.complete());
        case asrv1::TranscribeResponse::kPartialSegment:
        case asrv1::TranscribeResponse::kDocument:
        case asrv1::TranscribeResponse::EVENT_NOT_SET:
            return;
    }
}

void AsrDocumentFold::stamp_sources(
    google::protobuf::RepeatedPtrField<docv1::SourceType>* source, double start_seconds,
    double end_seconds, std::optional<float> avg_logprob) {
    docv1::TrackSource* track = source->Add()->mutable_track();
    track->set_start_time(start_seconds);
    // The upstream TrackSource validator requires end > start strictly, so
    // a zero-duration span (a keyframe instant, a degenerate segment) gets
    // a 1 ms epsilon here. The item's own provenance carries the true
    // instant unwidened, so nothing has to trust this number.
    track->set_end_time(end_seconds > start_seconds ? end_seconds : start_seconds + 0.001);
    // Every item this fold creates is attributable: additive merges with
    // other collectors' output rely on the tag to never collide silently.
    docv1::CollectorSource* collector = source->Add()->mutable_collector();
    collector->set_collector(kCollector);
    collector->set_model(model_);
    collector->set_version(version_);
    // A mean token logprob is at most 0 and exp() maps it onto the
    // schema's 0..1 range. That is a rescaled decoder score, not a
    // calibrated probability, so raw_score keeps the decoder's own number
    // verbatim and raw_score_kind names what it is; a consumer can rerank
    // without inverting the exp(). An absent score means the decoder
    // scored no token, and then neither field is written: no sentinel
    // stands in for a missing measurement. A present 0 is a real score.
    if (avg_logprob.has_value()) {
        double confidence = std::exp(static_cast<double>(*avg_logprob));
        collector->set_confidence(confidence > 1.0 ? 1.0 : confidence);
        collector->set_raw_score(static_cast<double>(*avg_logprob));
        collector->set_raw_score_kind(kRawScoreKind);
    }
}

void AsrDocumentFold::register_speaker(const std::string& speaker) {
    for (const std::string& known : document_.media().speakers()) {
        if (known == speaker) {
            return;
        }
    }
    document_.mutable_media()->add_speakers(speaker);
}

void AsrDocumentFold::on_media_info(const asrv1::MediaInfo& info) {
    auto* fields = document_.mutable_body()->mutable_meta()->mutable_custom_fields();
    set_string(fields, "asr.audio_codec", info.audio_codec());
    set_number(fields, "asr.sample_rate_hz", static_cast<double>(info.sample_rate_hz()));
    set_number(fields, "asr.channels", static_cast<double>(info.channels()));
    (*fields)["asr.has_video"].set_bool_value(info.has_video());
    if (info.has_video()) {
        set_string(fields, "asr.video_codec", info.video_codec());
    }
    if (info.duration_ms() != 0) {
        set_number(fields, "asr.declared_duration_ms", static_cast<double>(info.duration_ms()));
    }
    // The typed media facts the schema has slots for. The container's
    // declared duration seeds it; the trailer replaces it with what was
    // actually decoded, which is the authoritative number.
    docv1::MediaMeta* media = document_.mutable_media();
    if (!info.audio_codec().empty()) {
        media->set_codec(info.audio_codec());
    }
    if (info.duration_ms() != 0) {
        media->set_duration_ms(static_cast<double>(info.duration_ms()));
    }
}

void AsrDocumentFold::add_word_provenance(docv1::TextItemBase* base, const asrv1::Segment& segment,
                                          const std::string& text) {
    // Words arrive in reading order, so their charspans are found by
    // walking one cursor forward through the item text. A word the trim
    // moved or the decoder spelled differently simply gets no charspan;
    // its time span still stands.
    size_t byte_cursor = 0;
    int32_t point_cursor = 0;
    for (const asrv1::Word& word : segment.words()) {
        docv1::ProvenanceItem* prov = base->add_prov();
        time_span(prov, word.start_ms(), word.end_ms(), segment.speaker_id());
        const std::string needle = trimmed(word.text());
        if (needle.empty()) {
            continue;
        }
        size_t at = text.find(needle, byte_cursor);
        if (at == std::string::npos) {
            continue;
        }
        point_cursor += code_points(text, byte_cursor, at);
        docv1::IntSpan* charspan = prov->mutable_charspan();
        charspan->set_start(point_cursor);
        point_cursor += code_points(text, at, at + needle.size());
        charspan->set_end(point_cursor);
        byte_cursor = at + needle.size();
    }
}

void AsrDocumentFold::on_final_segment(const asrv1::Segment& segment) {
    std::string ref = "#/texts/" + std::to_string(document_.texts_size());
    docv1::TextItemBase* base = document_.add_texts()->mutable_text()->mutable_base();
    base->set_self_ref(ref);
    base->mutable_parent()->set_ref("#/body");
    base->set_label(docv1::DOC_ITEM_LABEL_TEXT);
    base->set_content_layer(docv1::CONTENT_LAYER_BODY);
    base->set_orig(segment.text());
    const std::string text = trimmed(segment.text());
    base->set_text(text);

    if (!segment.speaker_id().empty()) {
        register_speaker(segment.speaker_id());
    }

    // prov[0] locates the whole item; the word entries that may follow
    // locate its parts, each against this same text.
    docv1::ProvenanceItem* prov = base->add_prov();
    time_span(prov, segment.start_ms(), segment.end_ms(), segment.speaker_id());
    prov->mutable_charspan()->set_start(0);
    prov->mutable_charspan()->set_end(code_points(text, 0, text.size()));
    if (options_.word_provenance) {
        add_word_provenance(base, segment, text);
    }

    stamp_sources(base->mutable_source(), static_cast<double>(segment.start_ms()) / 1000.0,
                  static_cast<double>(segment.end_ms()) / 1000.0,
                  segment.has_avg_logprob() ? std::optional<float>(segment.avg_logprob())
                                            : std::nullopt);
    document_.mutable_body()->add_children()->set_ref(ref);
}

void AsrDocumentFold::on_keyframe(const asrv1::Keyframe& keyframe) {
    std::string ref = "#/pictures/" + std::to_string(document_.pictures_size());
    docv1::PictureItem* picture = document_.add_pictures();
    picture->set_self_ref(ref);
    picture->mutable_parent()->set_ref("#/body");
    picture->set_label(docv1::DOC_ITEM_LABEL_PICTURE);
    picture->set_content_layer(docv1::CONTENT_LAYER_BODY);
    docv1::ImageRef* image = picture->mutable_image();
    image->set_mimetype("image/png");
    image->mutable_size()->set_width(static_cast<double>(keyframe.width()));
    image->mutable_size()->set_height(static_cast<double>(keyframe.height()));
    // Pointer back into the typed stream, never embedded PNG bytes: the
    // Document is one gRPC message and must stay bounded.
    image->set_uri("keyframe:" + std::to_string(keyframe.timestamp_ms()));
    // The frame's true instant, as a zero-length span: the URI is a fetch
    // handle, not the carrier of when the frame is.
    time_span(picture->add_prov(), keyframe.timestamp_ms(), keyframe.timestamp_ms(), "");
    double seconds = static_cast<double>(keyframe.timestamp_ms()) / 1000.0;
    stamp_sources(picture->mutable_source(), seconds, seconds, std::nullopt);
    document_.mutable_body()->add_children()->set_ref(ref);
}

void AsrDocumentFold::on_complete(const asrv1::TranscriptComplete& complete) {
    docv1::BaseMeta* meta = document_.mutable_body()->mutable_meta();
    if (!complete.language().empty()) {
        // The detected language belongs in the slots the schema already
        // has: the enum for consumers that switch on it, the raw string
        // for codes the enum does not know, and the document-level tag.
        docv1::LanguageMetaField* language = meta->mutable_language();
        language->set_code(language_label(complete.language()));
        language->set_code_raw(complete.language());
        language->set_created_by(kCollector);
        document_.mutable_source_meta()->set_language(complete.language());
    }
    auto* fields = meta->mutable_custom_fields();
    set_number(fields, "asr.duration_ms", static_cast<double>(complete.duration_ms()));
    set_number(fields, "asr.token_count", static_cast<double>(complete.token_count()));
    if (complete.duration_ms() != 0) {
        document_.mutable_media()->set_duration_ms(static_cast<double>(complete.duration_ms()));
    }
    finished_ = true;
}

namespace {

// One structural view per item, whatever arena or oneof shape it lives in.
struct ItemView {
    std::string self_ref;
    std::string parent_ref;  // empty when the item has no parent (roots)
    const google::protobuf::RepeatedPtrField<docv1::RefItem>* children = nullptr;
};

void append_text_views(const docv1::Document& document, std::vector<ItemView>* views) {
    for (const docv1::BaseTextItem& item : document.texts()) {
        ItemView view;
        const docv1::TextItemBase* base = nullptr;
        switch (item.item_case()) {
            case docv1::BaseTextItem::kTitle: base = &item.title().base(); break;
            case docv1::BaseTextItem::kSectionHeader: base = &item.section_header().base(); break;
            case docv1::BaseTextItem::kListItem: base = &item.list_item().base(); break;
            case docv1::BaseTextItem::kFormula: base = &item.formula().base(); break;
            case docv1::BaseTextItem::kText: base = &item.text().base(); break;
            case docv1::BaseTextItem::kFieldHeading: base = &item.field_heading().base(); break;
            case docv1::BaseTextItem::kFieldValue: base = &item.field_value().base(); break;
            case docv1::BaseTextItem::kCode:
                // CodeItem inlines its base fields (see document.proto).
                view.self_ref = item.code().self_ref();
                if (item.code().has_parent()) {
                    view.parent_ref = item.code().parent().ref();
                }
                view.children = &item.code().children();
                views->push_back(view);
                continue;
            case docv1::BaseTextItem::ITEM_NOT_SET:
                views->push_back(view);  // empty self_ref reports as an error
                continue;
        }
        view.self_ref = base->self_ref();
        if (base->has_parent()) {
            view.parent_ref = base->parent().ref();
        }
        view.children = &base->children();
        views->push_back(view);
    }
}

}  // namespace

std::vector<std::string> document_integrity_errors(const docv1::Document& document) {
    std::vector<ItemView> views;
    views.push_back({document.body().self_ref(), "", &document.body().children()});
    views.push_back({document.furniture().self_ref(), "", &document.furniture().children()});
    for (const docv1::GroupItem& group : document.groups()) {
        ItemView view{group.self_ref(), "", &group.children()};
        if (group.has_parent()) {
            view.parent_ref = group.parent().ref();
        }
        views.push_back(view);
    }
    append_text_views(document, &views);
    for (const docv1::PictureItem& picture : document.pictures()) {
        ItemView view{picture.self_ref(), "", &picture.children()};
        if (picture.has_parent()) {
            view.parent_ref = picture.parent().ref();
        }
        views.push_back(view);
    }
    for (const docv1::TableItem& table : document.tables()) {
        ItemView view{table.self_ref(), "", &table.children()};
        if (table.has_parent()) {
            view.parent_ref = table.parent().ref();
        }
        views.push_back(view);
    }

    std::vector<std::string> errors;
    std::set<std::string> refs;
    std::map<std::string, std::set<std::string>> children_of;
    for (const ItemView& view : views) {
        if (view.self_ref.empty()) {
            errors.push_back("item with empty self_ref");
            continue;
        }
        if (!refs.insert(view.self_ref).second) {
            errors.push_back("duplicate self_ref " + view.self_ref);
        }
        if (view.children != nullptr) {
            for (const docv1::RefItem& child : *view.children) {
                children_of[view.self_ref].insert(child.ref());
            }
        }
    }
    for (const ItemView& view : views) {
        if (view.children != nullptr) {
            for (const docv1::RefItem& child : *view.children) {
                if (!refs.contains(child.ref())) {
                    errors.push_back(view.self_ref + " lists unresolvable child " + child.ref());
                }
            }
        }
        if (view.parent_ref.empty()) {
            continue;
        }
        if (!refs.contains(view.parent_ref)) {
            errors.push_back(view.self_ref + " has unresolvable parent " + view.parent_ref);
        } else if (!children_of[view.parent_ref].contains(view.self_ref)) {
            errors.push_back(view.parent_ref + " does not list child " + view.self_ref);
        }
    }
    return errors;
}

}  // namespace asr::doc
