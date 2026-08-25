// The Document fold in isolation: synthesized Transcribe events in, one
// structurally sound ai.pipestream.document.v1.Document out. No model
// weights and no server involved, so this runs on a bare checkout.

#include "document/document_fold.h"

#include <cmath>
#include <optional>
#include <print>
#include <string>

#include "fixture.h"

namespace asrv1 = ai::pipestream::asr::v1;
namespace docv1 = ai::pipestream::document::v1;

namespace {

asrv1::TranscribeResponse media_info_event() {
    asrv1::TranscribeResponse event;
    asrv1::MediaInfo* info = event.mutable_media_info();
    info->set_duration_ms(11000);
    info->set_audio_codec("wav");
    info->set_sample_rate_hz(16000);
    info->set_channels(1);
    info->set_has_video(true);
    info->set_video_codec("h264");
    return event;
}

asrv1::TranscribeResponse segment_event(bool final, uint32_t index, uint64_t start_ms,
                                        uint64_t end_ms, const std::string& text,
                                        std::optional<float> avg_logprob) {
    asrv1::TranscribeResponse event;
    asrv1::Segment* segment = final ? event.mutable_final_segment()
                                    : event.mutable_partial_segment();
    segment->set_index(index);
    segment->set_start_ms(start_ms);
    segment->set_end_ms(end_ms);
    segment->set_text(text);
    if (avg_logprob.has_value()) {
        segment->set_avg_logprob(*avg_logprob);
    }
    return event;
}

// Appends one word to the (final) segment the event carries.
void add_word(asrv1::TranscribeResponse& event, const std::string& text, uint64_t start_ms,
              uint64_t end_ms, float probability = 0.9F) {
    asrv1::Word* word = event.mutable_final_segment()->add_words();
    word->set_text(text);
    word->set_start_ms(start_ms);
    word->set_end_ms(end_ms);
    word->set_probability(probability);
}

asrv1::TranscribeResponse keyframe_event(uint64_t timestamp_ms) {
    asrv1::TranscribeResponse event;
    asrv1::Keyframe* frame = event.mutable_keyframe();
    frame->set_timestamp_ms(timestamp_ms);
    frame->set_width(640);
    frame->set_height(360);
    frame->set_png("not-a-real-png");
    return event;
}

asrv1::TranscribeResponse complete_event(const std::string& language = "en") {
    asrv1::TranscribeResponse event;
    asrv1::TranscriptComplete* complete = event.mutable_complete();
    complete->set_language(language);
    complete->set_duration_ms(11000);
    complete->set_segment_count(2);
    complete->set_token_count(20);
    complete->set_keyframe_count(1);
    return event;
}

// The item's text sliced by a charspan, treating the offsets as Unicode
// code points: the test decodes independently of the fold so a byte-offset
// regression cannot pass.
std::string code_point_slice(const std::string& text, const docv1::IntSpan& span) {
    std::string out;
    int32_t point = 0;
    for (size_t i = 0; i < text.size(); i++) {
        bool starts_point = (static_cast<unsigned char>(text[i]) & 0xC0) != 0x80;
        if (starts_point) {
            if (point == span.end()) {
                break;
            }
            point++;
        }
        if (point > span.start()) {
            out += text[i];
        }
    }
    return out;
}

void verify_fold() {
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    fold.set_source("address.wav", "audio/wav", 0xFEEDFACEULL);
    fold.consume(media_info_event());
    fold.consume(segment_event(/*final=*/false, 0, 0, 4000, " partial noise", -0.4F));
    fold.consume(segment_event(/*final=*/true, 0, 0, 5000, " And so my fellow Americans",
                               -0.25F));
    fold.consume(keyframe_event(5000));
    fold.consume(segment_event(/*final=*/true, 1, 5000, 11000,
                               " ask not what your country can do for you", -0.5F));
    require(!fold.finished(), "not finished before the trailer");
    fold.consume(complete_event());
    require(fold.finished(), "finished after the trailer");

    docv1::Document document = fold.take();
    std::vector<std::string> errors = asr::doc::document_integrity_errors(document);
    require(errors.empty(), "integrity: " + (errors.empty() ? "" : errors.front()));

    require(document.schema_name() == "docling_document_v2", "schema name");
    require(document.texts_size() == 2, "finals fold, partials do not");
    require(document.pictures_size() == 1, "keyframe folds to a picture");
    require(document.body().children_size() == 3, "body children: text, picture, text");
    // Arrival order is wire order: final 0, keyframe, final 1.
    require(document.body().children(0).ref() == "#/texts/0" &&
                document.body().children(1).ref() == "#/pictures/0" &&
                document.body().children(2).ref() == "#/texts/1",
            "body children in arrival order");

    // The source identity the event stream does not carry.
    require(document.name() == "address.wav", "document is named after the media");
    require(document.origin().filename() == "address.wav" &&
                document.origin().mimetype() == "audio/wav" &&
                document.origin().binary_hash() == 0xFEEDFACEULL,
            "origin carries filename, mimetype, and content hash");

    const docv1::TextItemBase& first = document.texts(0).text().base();
    require(first.orig() == " And so my fellow Americans", "orig keeps the verbatim text");
    require(first.text() == "And so my fellow Americans", "text is trimmed");
    require(first.source_size() == 2, "track + collector sources");
    require(first.source(0).track().start_time() == 0.0 &&
                first.source(0).track().end_time() == 5.0,
            "track range in seconds");
    const docv1::CollectorSource& collector = first.source(1).collector();
    require(collector.collector() == "asr" && collector.model() == "tiny.en" &&
                collector.version() == "1.2.3",
            "collector attribution");
    require(std::abs(collector.confidence() - std::exp(-0.25)) < 1e-9,
            "confidence rescales avg_logprob into 0..1");
    require(std::abs(first.meta().custom_fields().at("pipestream__avg_logprob").number_value() +
                     0.25) < 1e-6,
            "the raw decoder score rides the item meta, unrescaled");

    // Time is provenance: one entry for the segment, no words on this run.
    require(first.prov_size() == 1, "segment provenance, no word entries without words");
    const docv1::ProvenanceItem& prov = first.prov(0);
    require(prov.has_time() && prov.time().start_ms() == 0.0 && prov.time().end_ms() == 5000.0,
            "segment time span in milliseconds");
    require(prov.page_no() == 0 && !prov.has_bbox(), "no invented page or box on media items");
    require(prov.charspan().start() == 0 &&
                prov.charspan().end() == static_cast<int32_t>(first.text().size()),
            "segment charspan covers the whole item text");
    require(!prov.time().has_speaker(), "no speaker claimed without diarization");

    const docv1::PictureItem& picture = document.pictures(0);
    require(picture.image().uri() == "keyframe:5000",
            "picture points at the typed keyframe event, no embedded bytes");
    require(picture.image().mimetype() == "image/png" &&
                picture.image().size().width() == 640.0 &&
                picture.image().size().height() == 360.0,
            "image ref carries the frame facts");
    require(picture.source_size() == 2 && picture.source(0).track().start_time() == 5.0 &&
                picture.source(0).track().end_time() == 5.001,
            "keyframe track keeps the 1 ms zero-duration epsilon the validator wants");
    require(picture.prov_size() == 1 && picture.prov(0).time().start_ms() == 5000.0 &&
                picture.prov(0).time().end_ms() == 5000.0,
            "keyframe provenance is the true instant, unwidened");
    require(!picture.source(1).collector().has_confidence(),
            "keyframes claim no model confidence");

    const docv1::BaseMeta& meta = document.body().meta();
    require(meta.language().code() == docv1::HUMAN_LANGUAGE_LABEL_EN,
            "the ISO code lands in the language enum");
    require(meta.language().code_raw() == "en" && meta.language().created_by() == "asr",
            "raw code and producer kept alongside");
    require(document.source_meta().language() == "en", "document meta names the language too");
    require(meta.custom_fields().at("asr.audio_codec").string_value() == "wav" &&
                meta.custom_fields().at("asr.has_video").bool_value() &&
                meta.custom_fields().at("asr.duration_ms").number_value() == 11000.0,
            "media facts in body custom fields");
    require(document.media().duration_ms() == 11000.0 && document.media().codec() == "wav",
            "typed media meta carries the decoded duration and codec");
    require(document.media().speakers_size() == 0, "no speakers registered without diarization");
}

void verify_word_provenance() {
    // Multi-byte text with a repeated word: charspans must be code points,
    // and the second "sí" must not match the first one's offset.
    const std::string text = " Café über sí sí";
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    asrv1::TranscribeResponse event = segment_event(/*final=*/true, 0, 1000, 4000, text, -0.2F);
    add_word(event, " Café", 1000, 1500, 0.91F);
    add_word(event, " über", 1500, 2200, 0.72F);
    add_word(event, " sí", 2200, 3000, 0.63F);
    add_word(event, " sí", 3000, 4000, 0.55F);
    fold.consume(event);

    const docv1::TextItemBase& base = fold.document().texts(0).text().base();
    const std::string trimmed = base.text();
    require(trimmed == "Café über sí sí", "text is trimmed, words are not");
    require(base.prov_size() == 5, "one segment entry plus one per word");
    require(base.prov(0).time().start_ms() == 1000.0 && base.prov(0).time().end_ms() == 4000.0,
            "entry 0 is the segment span");
    require(base.prov(0).charspan().end() == 15, "segment charspan counts code points, not bytes");

    const int32_t starts[] = {0, 5, 10, 13};
    const int32_t ends[] = {4, 9, 12, 15};
    const std::string words[] = {"Café", "über", "sí", "sí"};
    const double word_starts[] = {1000.0, 1500.0, 2200.0, 3000.0};
    for (int i = 0; i < 4; i++) {
        const docv1::ProvenanceItem& prov = base.prov(i + 1);
        require(prov.has_time() && prov.time().start_ms() == word_starts[i],
                "word " + words[i] + " keeps its own start");
        require(prov.time().end_ms() > prov.time().start_ms(),
                "word " + words[i] + " spans forward");
        require(prov.charspan().start() == starts[i] && prov.charspan().end() == ends[i],
                "word " + words[i] + " charspan in code points");
        require(code_point_slice(trimmed, prov.charspan()) == words[i],
                "charspan slices back to " + words[i]);
    }
}

void verify_word_provenance_can_be_turned_off() {
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3",
                                   asr::doc::FoldOptions{.word_provenance = false});
    asrv1::TranscribeResponse event =
        segment_event(/*final=*/true, 0, 0, 2000, " one two", -0.2F);
    add_word(event, " one", 0, 900);
    add_word(event, " two", 900, 2000);
    fold.consume(event);
    const docv1::TextItemBase& base = fold.document().texts(0).text().base();
    require(base.prov_size() == 1, "only the segment entry when word provenance is off");
    require(base.prov(0).time().end_ms() == 2000.0, "the segment span still stands");
}

void verify_speaker_registry() {
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    auto turn = [](uint32_t index, uint64_t start_ms, uint64_t end_ms, const std::string& text,
                   const std::string& speaker) {
        asrv1::TranscribeResponse event =
            segment_event(/*final=*/true, index, start_ms, end_ms, text, -0.2F);
        event.mutable_final_segment()->set_speaker_id(speaker);
        return event;
    };
    asrv1::TranscribeResponse first = turn(0, 0, 1000, " hello there", "S1");
    add_word(first, " hello", 0, 500);
    add_word(first, " there", 500, 1000);
    fold.consume(first);
    fold.consume(turn(1, 1000, 2000, " hi back", "S2"));
    fold.consume(turn(2, 2000, 3000, " indeed", "S1"));
    fold.consume(complete_event());

    docv1::Document document = fold.take();
    require(document.media().speakers_size() == 2, "distinct speakers only");
    require(document.media().speakers(0) == "S1" && document.media().speakers(1) == "S2",
            "speakers registered in first-appearance order");
    require(document.texts(0).text().base().prov(0).time().speaker() == "S1" &&
                document.texts(1).text().base().prov(0).time().speaker() == "S2" &&
                document.texts(2).text().base().prov(0).time().speaker() == "S1",
            "every segment span names its speaker");
    require(document.texts(0).text().base().prov(1).time().speaker() == "S1" &&
                document.texts(0).text().base().prov(2).time().speaker() == "S1",
            "word spans inherit the segment's speaker");
}

void verify_language_slots() {
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    fold.consume(segment_event(/*final=*/true, 0, 0, 1000, " bonjour", -0.2F));
    fold.consume(complete_event("fr"));
    require(fold.document().body().meta().language().code() == docv1::HUMAN_LANGUAGE_LABEL_FR,
            "fr maps onto the enum");
    require(fold.document().source_meta().language() == "fr", "document meta language");

    // A code the enum does not know keeps the raw string and claims no
    // enum value rather than guessing one.
    asr::doc::AsrDocumentFold exotic("tiny.en", "1.2.3");
    exotic.consume(segment_event(/*final=*/true, 0, 0, 1000, " nei hou", -0.2F));
    exotic.consume(complete_event("yue"));
    const docv1::LanguageMetaField& language = exotic.document().body().meta().language();
    require(language.code() == docv1::HUMAN_LANGUAGE_LABEL_UNSPECIFIED,
            "an unknown code leaves the enum unspecified");
    require(language.code_raw() == "yue", "the raw code survives regardless");
}

void verify_confidence_presence() {
    // No score at all: nothing is claimed.
    asr::doc::AsrDocumentFold silent("tiny.en", "1.2.3");
    silent.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", std::nullopt));
    const docv1::TextItemBase& unscored = silent.document().texts(0).text().base();
    require(!unscored.source(1).collector().has_confidence(),
            "an absent avg_logprob claims no confidence");
    require(!unscored.meta().custom_fields().contains("pipestream__avg_logprob"),
            "and records no raw score");

    // A real zero: the decoder was certain, which is a value, not a gap.
    asr::doc::AsrDocumentFold certain("tiny.en", "1.2.3");
    certain.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", 0.0F));
    const docv1::TextItemBase& scored = certain.document().texts(0).text().base();
    require(scored.source(1).collector().has_confidence(),
            "avg_logprob 0 is a score, not a missing one");
    require(std::abs(scored.source(1).collector().confidence() - 1.0) < 1e-9,
            "logprob 0 rescales to confidence 1");
    require(scored.meta().custom_fields().at("pipestream__avg_logprob").number_value() == 0.0,
            "the raw zero is recorded");
}

void verify_integrity_checker_catches_breakage() {
    // The checker must actually detect asymmetry, or the fold tests above
    // prove nothing.
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    fold.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", -0.5F));
    fold.consume(complete_event());
    docv1::Document document = fold.take();
    document.mutable_body()->mutable_children(0)->set_ref("#/texts/9");
    require(!asr::doc::document_integrity_errors(document).empty(),
            "dangling child ref detected");

    asr::doc::AsrDocumentFold fold2("tiny.en", "1.2.3");
    fold2.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", -0.5F));
    fold2.consume(complete_event());
    docv1::Document document2 = fold2.take();
    document2.mutable_body()->clear_children();
    require(!asr::doc::document_integrity_errors(document2).empty(),
            "parent not listing its child detected");
}

}  // namespace

int main() {
    try {
        verify_fold();
        verify_word_provenance();
        verify_word_provenance_can_be_turned_off();
        verify_speaker_registry();
        verify_language_slots();
        verify_confidence_presence();
        verify_integrity_checker_catches_breakage();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("document-fold-test passed");
    return 0;
}
