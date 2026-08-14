// The Document fold in isolation: synthesized Transcribe events in, one
// structurally sound ai.pipestream.document.v1.Document out. No model
// weights and no server involved, so this runs on a bare checkout.

#include "document/document_fold.h"

#include <cmath>
#include <iostream>

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
                                        float avg_logprob) {
    asrv1::TranscribeResponse event;
    asrv1::Segment* segment = final ? event.mutable_final_segment()
                                    : event.mutable_partial_segment();
    segment->set_index(index);
    segment->set_start_ms(start_ms);
    segment->set_end_ms(end_ms);
    segment->set_text(text);
    segment->set_avg_logprob(avg_logprob);
    return event;
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

asrv1::TranscribeResponse complete_event() {
    asrv1::TranscribeResponse event;
    asrv1::TranscriptComplete* complete = event.mutable_complete();
    complete->set_language("en");
    complete->set_duration_ms(11000);
    complete->set_segment_count(2);
    complete->set_token_count(20);
    complete->set_keyframe_count(1);
    return event;
}

void verify_fold() {
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
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

    require(document.schema_name() == "docling_document_v2", "docling schema name");
    require(document.texts_size() == 2, "finals fold, partials do not");
    require(document.pictures_size() == 1, "keyframe folds to a picture");
    require(document.body().children_size() == 3, "body children: text, picture, text");
    // Arrival order is wire order: final 0, keyframe, final 1.
    require(document.body().children(0).ref() == "#/texts/0" &&
                document.body().children(1).ref() == "#/pictures/0" &&
                document.body().children(2).ref() == "#/texts/1",
            "body children in arrival order");

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
            "confidence is exp(avg_logprob)");
    require(first.prov_size() == 0, "no invented page provenance");

    const docv1::PictureItem& picture = document.pictures(0);
    require(picture.image().uri() == "keyframe:5000",
            "picture points at the typed keyframe event, no embedded bytes");
    require(picture.image().mimetype() == "image/png" &&
                picture.image().size().width() == 640.0 &&
                picture.image().size().height() == 360.0,
            "image ref carries the frame facts");
    require(picture.source_size() == 2 && picture.source(0).track().start_time() == 5.0 &&
                picture.source(0).track().end_time() == 5.001,
            "keyframe track gets docling's 1 ms zero-duration epsilon");
    require(!picture.source(1).collector().has_confidence(),
            "keyframes claim no model confidence");

    const docv1::BaseMeta& meta = document.body().meta();
    require(meta.language().code_raw() == "en", "language folded from the trailer");
    require(meta.custom_fields().at("asr.audio_codec").string_value() == "wav" &&
                meta.custom_fields().at("asr.has_video").bool_value() &&
                meta.custom_fields().at("asr.duration_ms").number_value() == 11000.0,
            "media facts in body custom fields");
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

void verify_unknown_logprob_omits_confidence() {
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    fold.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", 0.0F));
    require(!fold.document().texts(0).text().base().source(1).collector().has_confidence(),
            "avg_logprob 0 means unknown, so no confidence claim");
}

}  // namespace

int main() {
    try {
        verify_fold();
        verify_integrity_checker_catches_breakage();
        verify_unknown_logprob_omits_confidence();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "document-fold-test passed\n";
    return 0;
}
