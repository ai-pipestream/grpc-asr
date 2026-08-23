#include "document/document_fold.h"

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

}  // namespace

AsrDocumentFold::AsrDocumentFold(std::string model, std::string version)
    : model_(std::move(model)), version_(std::move(version)) {
    document_.set_schema_name(kSchemaName);
    docv1::GroupItem* body = document_.mutable_body();
    body->set_self_ref("#/body");
    body->set_content_layer(docv1::CONTENT_LAYER_BODY);
    docv1::GroupItem* furniture = document_.mutable_furniture();
    furniture->set_self_ref("#/furniture");
    furniture->set_content_layer(docv1::CONTENT_LAYER_FURNITURE);
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
    double end_seconds, float avg_logprob) {
    docv1::TrackSource* track = source->Add()->mutable_track();
    track->set_start_time(start_seconds);
    // Docling's TrackSource validator requires end > start strictly, so a
    // zero-duration span (a keyframe instant, a degenerate segment) gets the
    // same 1 ms epsilon docling itself applies (ZERO_DURATION_SEGMENT_EPS in
    // the ASR transcriber, timestamp + 0.001 in the video pipeline).
    track->set_end_time(end_seconds > start_seconds ? end_seconds : start_seconds + 0.001);
    // Every item this fold creates is attributable: additive merges with
    // other collectors' output rely on the tag to never collide silently.
    docv1::CollectorSource* collector = source->Add()->mutable_collector();
    collector->set_collector(kCollector);
    collector->set_model(model_);
    collector->set_version(version_);
    // avg logprob 0 means "unknown" on the wire; a real mean token
    // logprob is negative and exp() maps it onto the schema's 0..1 range
    // (the vlm-convert precedent).
    if (avg_logprob != 0.0F) {
        double confidence = std::exp(static_cast<double>(avg_logprob));
        collector->set_confidence(confidence > 1.0 ? 1.0 : confidence);
    }
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
}

void AsrDocumentFold::on_final_segment(const asrv1::Segment& segment) {
    std::string ref = "#/texts/" + std::to_string(document_.texts_size());
    docv1::TextItemBase* base = document_.add_texts()->mutable_text()->mutable_base();
    base->set_self_ref(ref);
    base->mutable_parent()->set_ref("#/body");
    base->set_label(docv1::DOC_ITEM_LABEL_TEXT);
    base->set_content_layer(docv1::CONTENT_LAYER_BODY);
    base->set_orig(segment.text());
    base->set_text(trimmed(segment.text()));
    stamp_sources(base->mutable_source(), static_cast<double>(segment.start_ms()) / 1000.0,
                  static_cast<double>(segment.end_ms()) / 1000.0, segment.avg_logprob());
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
    double seconds = static_cast<double>(keyframe.timestamp_ms()) / 1000.0;
    stamp_sources(picture->mutable_source(), seconds, seconds, 0.0F);
    document_.mutable_body()->add_children()->set_ref(ref);
}

void AsrDocumentFold::on_complete(const asrv1::TranscriptComplete& complete) {
    docv1::BaseMeta* meta = document_.mutable_body()->mutable_meta();
    if (!complete.language().empty()) {
        meta->mutable_language()->set_code_raw(complete.language());
    }
    auto* fields = meta->mutable_custom_fields();
    set_number(fields, "asr.duration_ms", static_cast<double>(complete.duration_ms()));
    set_number(fields, "asr.token_count", static_cast<double>(complete.token_count()));
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
