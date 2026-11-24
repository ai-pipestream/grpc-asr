// The Document fold in isolation: synthesized Transcribe events in, one
// structurally sound ai.pipestream.document.v1.Document out. No model
// weights and no server involved, so this runs on a bare checkout.

#include "document/document_fold.h"

#include <cmath>
#include <optional>
#include <print>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/util/message_differencer.h>

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
    info->set_probe_state(asrv1::MEDIA_PROBE_STATE_PROBED);
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
    // Ordered by media time, with pictures before texts at equal start_ms
    // (5000 ms): text0 (0..5000), picture0 (5000), text1 (5000..11000).
    require(document.body().children(0).ref() == "#/texts/0" &&
                document.body().children(1).ref() == "#/pictures/0" &&
                document.body().children(2).ref() == "#/texts/1",
            "body children in media time order");

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
    require(collector.has_raw_score() && collector.raw_score() < 0.0,
            "the raw decoder score is a typed double on the collector source");
    require(std::abs(collector.raw_score() + 0.25) < 1e-6,
            "raw_score is the avg logprob verbatim, unrescaled");
    require(collector.raw_score_kind() == "avg_logprob", "raw_score_kind names the signal");
    require(!first.meta().custom_fields().contains("pipestream__avg_logprob"),
            "the retired meta custom field is gone");

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
    require(!picture.source(1).collector().has_raw_score() &&
                !picture.source(1).collector().has_raw_score_kind(),
            "and no raw score either: a keyframe is not something the decoder scored");

    const docv1::BaseMeta& meta = document.body().meta();
    require(meta.language().code() == docv1::HUMAN_LANGUAGE_LABEL_EN,
            "the ISO code lands in the language enum");
    require(meta.language().code_raw() == "en" && meta.language().created_by() == "asr",
            "raw code and producer kept alongside");
    require(document.source_meta().language() == "en", "document meta names the language too");
    // Every media fact has a typed slot on Document.media (#40): none of
    // them goes into the untyped custom_fields bag, under any key.
    const docv1::MediaMeta& media = document.media();
    require(media.codec() == "wav" && media.has_sample_rate_hz() &&
                media.sample_rate_hz() == 16000 && media.has_channels() && media.channels() == 1,
            "the audio stream's codec, rate and channels are typed media facts");
    require(media.has_has_video() && media.has_video() && media.video_codec() == "h264",
            "so are the video flag and the video codec");
    require(media.has_declared_duration_ms() && media.declared_duration_ms() == 11000.0,
            "the duration the container states has its own slot");
    require(media.has_duration_ms() && media.duration_ms() == 11000.0,
            "duration_ms is what the trailer says was decoded");
    require(media.has_token_count() && media.token_count() == 20,
            "the trailer's token count is typed too");
    require(meta.custom_fields().empty(),
            "no media fact is left in body custom fields, got " +
                std::to_string(meta.custom_fields().size()) + " key(s)" +
                (meta.custom_fields().empty() ? "" : ", first " +
                                                         meta.custom_fields().begin()->first));
    require(document.media().speakers_size() == 0, "no speakers registered without diarization");
}

// Media that was never probed has no sample rate, channel count or video
// flag to report. The fold must not write the zeros MediaInfo carries in
// their place into the Document as if they were facts (#37): a stereo call
// recording is not "0 channels", and a video is not "has_video: false".
void verify_unprobed_media_facts_are_left_out() {
    auto fold_of = [](asrv1::MediaProbeState state) {
        asrv1::TranscribeResponse event;
        asrv1::MediaInfo* info = event.mutable_media_info();
        info->set_audio_codec("opus");
        info->set_container(asrv1::CONTAINER_FAMILY_MATROSKA);
        info->set_probe_state(state);
        if (state == asrv1::MEDIA_PROBE_STATE_PROBED) {
            info->set_sample_rate_hz(48000);
            info->set_channels(2);
        }
        asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
        fold.set_source("call.webm", "video/webm", 0x1234ULL);
        fold.consume(event);
        fold.consume(segment_event(/*final=*/true, 0, 0, 5000, " hello", -0.25F));
        fold.consume(complete_event());
        return fold.take();
    };

    for (const asrv1::MediaProbeState unknown :
         {asrv1::MEDIA_PROBE_STATE_NOT_PROBED, asrv1::MEDIA_PROBE_STATE_UNSPECIFIED}) {
        const docv1::Document unprobed = fold_of(unknown);
        const docv1::MediaMeta& media = unprobed.media();
        require(!media.has_sample_rate_hz() && !media.has_channels() && !media.has_has_video() &&
                    !media.has_video_codec(),
                "stream facts are not written for media that was not probed");
        require(media.codec() == "opus", "the sniffed codec is still a fact and still recorded");
        require(!media.has_declared_duration_ms(),
                "a container that stated no duration leaves the declared slot unset");
        require(unprobed.body().meta().custom_fields().empty(),
                "and nothing about the media is in body custom fields");
    }

    const docv1::Document probed = fold_of(asrv1::MEDIA_PROBE_STATE_PROBED);
    const docv1::MediaMeta& media = probed.media();
    require(media.sample_rate_hz() == 48000 && media.channels() == 2 && media.has_has_video() &&
                !media.has_video(),
            "probed media keeps its facts, a real false included");
    require(!media.has_video_codec(), "audio-only media names no video codec");
    require(probed.body().meta().custom_fields().empty(),
            "probed media leaves nothing in body custom fields either");
}

// duration_ms is the decoded length and nothing else. The container's own
// claim has declared_duration_ms, so a trailer that decoded nothing must
// not leave the claim standing in the decoded slot. A token count of zero
// beside transcript text is a count nobody took (a remote endpoint that
// returned no token lists), and is left unset instead of written as 0.
void verify_media_measurements_are_not_made_up() {
    asrv1::TranscribeResponse trailer = complete_event();
    trailer.mutable_complete()->set_duration_ms(0);
    trailer.mutable_complete()->set_token_count(0);

    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    fold.consume(media_info_event());
    fold.consume(segment_event(/*final=*/true, 0, 0, 5000, " hello", -0.25F));
    fold.consume(trailer);
    const docv1::Document document = fold.take();
    require(document.media().declared_duration_ms() == 11000.0,
            "the declared duration is recorded as declared");
    require(!document.media().has_duration_ms(),
            "the declared duration does not stand in for a decoded one");
    require(!document.media().has_token_count(),
            "zero tokens beside transcript text is not a count");

    asr::doc::AsrDocumentFold silent("tiny.en", "1.2.3");
    silent.consume(media_info_event());
    silent.consume(trailer);
    const docv1::Document nothing_said = silent.take();
    require(nothing_said.media().has_token_count() && nothing_said.media().token_count() == 0,
            "a transcript with no text really has zero tokens");
}

// What TranscriptComplete.decode_warnings says about the media is also a
// fact about the Document folded from it: a transcript with a hole in it
// must not arrive downstream looking whole. Each warning becomes one typed
// CollectorWarning attributed to this collector.
void verify_decode_warnings_fold_into_document_warnings() {
    asrv1::TranscribeResponse trailer = complete_event();
    asrv1::DecodeWarning* errors = trailer.mutable_complete()->add_decode_warnings();
    errors->set_code(asrv1::DECODE_WARNING_CODE_DECODER_ERRORS);
    errors->set_error_count(3);
    errors->set_message("[matroska,webm] Read error");
    asrv1::DecodeWarning* cut = trailer.mutable_complete()->add_decode_warnings();
    cut->set_code(asrv1::DECODE_WARNING_CODE_SHORTER_THAN_DECLARED);
    cut->set_declared_duration_ms(22000);
    cut->set_decoded_duration_ms(11000);
    cut->set_message("decoded 11000 ms of audio where the media states 22000 ms");

    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    fold.consume(media_info_event());
    fold.consume(segment_event(/*final=*/true, 0, 0, 5000, " hello", -0.25F));
    fold.consume(trailer);
    const docv1::Document document = fold.take();
    require(document.warnings_size() == 2,
            "one Document warning per decode warning, got " +
                std::to_string(document.warnings_size()));

    const docv1::CollectorWarning& malformed = document.warnings(0);
    require(malformed.code() == docv1::WARNING_CODE_SOURCE_STRUCTURE_MALFORMED,
            "decoder errors are a source that breaks its own format");
    require(malformed.has_count() && malformed.count() == 3,
            "the entry stands for every error line the decoder printed");
    require(malformed.message().find("[matroska,webm] Read error") != std::string::npos,
            "the decoder's own first complaint stays in the words: " + malformed.message());
    require(malformed.source().collector() == "asr" && malformed.source().model() == "tiny.en" &&
                malformed.source().version() == "1.2.3",
            "the warning names the collector that raised it");
    require(!malformed.has_page_no() && !malformed.has_item_ref(),
            "media has no page, and the warning is about no single item");

    const docv1::CollectorWarning& incomplete = document.warnings(1);
    require(incomplete.code() == docv1::WARNING_CODE_SOURCE_INPUT_INCOMPLETE,
            "audio shorter than the media states is an incomplete source");
    require(!incomplete.has_count(), "one finding, so no count");
    require(incomplete.message() == cut->message(), "the service's own words are kept");
    require(incomplete.source().collector() == "asr", "attributed to this collector as well");

    // A code this build does not know still arrives, with its words and no
    // typed meaning claimed for it.
    asrv1::TranscribeResponse newer = complete_event();
    asrv1::DecodeWarning* unknown = newer.mutable_complete()->add_decode_warnings();
    unknown->set_code(static_cast<asrv1::DecodeWarningCode>(99));
    unknown->set_message("something a later server learned to say");
    asr::doc::AsrDocumentFold later("tiny.en", "1.2.3");
    later.consume(newer);
    const docv1::Document passed_on = later.take();
    require(passed_on.warnings_size() == 1 &&
                passed_on.warnings(0).code() == docv1::WARNING_CODE_UNSPECIFIED &&
                passed_on.warnings(0).message() == "something a later server learned to say",
            "an unknown decode warning keeps its words under WARNING_CODE_UNSPECIFIED");

    asr::doc::AsrDocumentFold clean("tiny.en", "1.2.3");
    clean.consume(media_info_event());
    clean.consume(complete_event());
    require(clean.take().warnings_size() == 0, "a clean decode folds to no warnings");
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

void verify_score_presence() {
    // No score at all: nothing is claimed, and nothing stands in for it.
    asr::doc::AsrDocumentFold silent("tiny.en", "1.2.3");
    silent.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", std::nullopt));
    const docv1::TextItemBase& unscored = silent.document().texts(0).text().base();
    const docv1::CollectorSource& unscored_collector = unscored.source(1).collector();
    require(!unscored_collector.has_confidence(), "an absent avg_logprob claims no confidence");
    require(!unscored_collector.has_raw_score(), "and no raw score");
    require(!unscored_collector.has_raw_score_kind(), "and names no raw score kind");
    require(unscored_collector.raw_score() == 0.0 && unscored_collector.raw_score_kind().empty(),
            "the unset default is the proto default, never a written sentinel");
    require(unscored.meta().custom_fields().empty(),
            "an unscored item carries no meta custom fields at all");

    // A real zero: the decoder was certain, which is a value, not a gap.
    asr::doc::AsrDocumentFold certain("tiny.en", "1.2.3");
    certain.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", 0.0F));
    const docv1::CollectorSource& certain_collector =
        certain.document().texts(0).text().base().source(1).collector();
    require(certain_collector.has_confidence(), "avg_logprob 0 is a score, not a missing one");
    require(std::abs(certain_collector.confidence() - 1.0) < 1e-9,
            "logprob 0 rescales to confidence 1");
    require(certain_collector.has_raw_score() && certain_collector.raw_score() == 0.0,
            "the raw zero is recorded as a set field, not left to the default");
    require(certain_collector.raw_score_kind() == "avg_logprob",
            "a zero score still names its kind");

    // The two are independent readings of the same measurement: raw_score
    // stays the decoder's own number while confidence stays the rescale,
    // so a consumer can invert neither into the other by accident.
    asr::doc::AsrDocumentFold unsure("tiny.en", "1.2.3");
    unsure.consume(segment_event(/*final=*/true, 0, 0, 1000, "hello", -1.75F));
    const docv1::CollectorSource& unsure_collector =
        unsure.document().texts(0).text().base().source(1).collector();
    require(std::abs(unsure_collector.raw_score() + 1.75) < 1e-6,
            "raw_score is uncalibrated and may sit far outside 0..1");
    require(std::abs(unsure_collector.confidence() - std::exp(-1.75)) < 1e-9,
            "confidence is derived exactly as before, unchanged by the raw score");
    require(unsure_collector.confidence() != unsure_collector.raw_score(),
            "the rescaled and raw readings are distinct numbers");
    require(!unsure.document().texts(0).text().base().meta().custom_fields().contains(
                "pipestream__avg_logprob"),
            "no item writes the retired custom field, whatever it scored");
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

void verify_time_ordered_body_with_early_keyframes() {
    // Keyframe child finishes ahead of transcription on long media: arenas
    // keep arrival order so refs never shift, while body.children orders
    // by prov[0].time.start_ms (pictures before texts at equal start_ms).
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    fold.consume(keyframe_event(5000));
    fold.consume(keyframe_event(15000));
    fold.consume(segment_event(/*final=*/true, 0, 0, 4000, " intro", -0.2F));
    fold.consume(segment_event(/*final=*/true, 1, 10000, 14000, " middle", -0.2F));
    fold.consume(complete_event());

    docv1::Document document = fold.take();
    std::vector<std::string> errors = asr::doc::document_integrity_errors(document);
    require(errors.empty(), "integrity after time reorder: " + (errors.empty() ? "" : errors.front()));
    require(document.pictures(0).image().uri() == "keyframe:5000" &&
                document.pictures(1).image().uri() == "keyframe:15000",
            "pictures arena stays in arrival order");
    require(document.texts(0).text().base().text() == "intro" &&
                document.texts(1).text().base().text() == "middle",
            "texts arena stays in arrival order");
    require(document.body().children_size() == 4, "four children in body");
    require(document.body().children(0).ref() == "#/texts/0" &&
                document.body().children(1).ref() == "#/pictures/0" &&
                document.body().children(2).ref() == "#/texts/1" &&
                document.body().children(3).ref() == "#/pictures/1",
            "body.children ordered by media time: text0, picture0, text1, picture1");

    // Equal timestamp tie-breaker: segment arriving BEFORE keyframe at the
    // same start_ms still places the picture before the text.
    asr::doc::AsrDocumentFold tie("tiny.en", "1.2.3");
    tie.consume(segment_event(/*final=*/true, 0, 5000, 8000, " spoken over slide", -0.2F));
    tie.consume(keyframe_event(5000));
    tie.consume(complete_event());
    docv1::Document tie_doc = tie.take();
    require(asr::doc::document_integrity_errors(tie_doc).empty(), "tie integrity");
    require(tie_doc.body().children(0).ref() == "#/pictures/0" &&
                tie_doc.body().children(1).ref() == "#/texts/0",
            "picture precedes text at equal start_ms");
}

void verify_suppressed_segment_skipped() {
    asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
    asrv1::TranscribeResponse suppressed =
        segment_event(/*final=*/true, 0, 0, 3000, " thank you for watching", -1.8F);
    suppressed.mutable_final_segment()->set_no_speech_prob(0.92F);
    suppressed.mutable_final_segment()->set_suppressed(true);
    suppressed.mutable_final_segment()->set_speaker_id("C2");
    fold.consume(suppressed);

    asrv1::TranscribeResponse real =
        segment_event(/*final=*/true, 1, 3000, 6000, " actual speech", -0.2F);
    real.mutable_final_segment()->set_speaker_id("C1");
    fold.consume(real);
    fold.consume(complete_event());

    docv1::Document document = fold.take();
    require(asr::doc::document_integrity_errors(document).empty(), "integrity with suppressed");
    require(document.texts_size() == 1, "suppressed segment produces no text item");
    require(document.texts(0).text().base().text() == "actual speech",
            "non-suppressed segment survives");
    require(document.media().speakers_size() == 1 && document.media().speakers(0) == "C1",
            "suppressed segment does not register a phantom speaker");
}

void verify_multilingual_fold() {
    asr::doc::AsrDocumentFold fold("base", "1.2.3");
    asrv1::TranscribeResponse seg0 = segment_event(/*final=*/true, 0, 0, 4000, " hello", -0.2F);
    seg0.mutable_final_segment()->set_language("en");
    asrv1::TranscribeResponse seg1 = segment_event(/*final=*/true, 1, 4000, 8000, " hola", -0.2F);
    seg1.mutable_final_segment()->set_language("es");
    fold.consume(seg0);
    fold.consume(seg1);

    asrv1::TranscribeResponse trailer = complete_event("en");
    trailer.mutable_complete()->add_languages("en");
    trailer.mutable_complete()->add_languages("es");
    fold.consume(trailer);

    docv1::Document document = fold.take();
    require(asr::doc::document_integrity_errors(document).empty(), "multilingual integrity");
    const auto& lang0 = document.texts(0).text().base().meta().language();
    const auto& lang1 = document.texts(1).text().base().meta().language();
    require(lang0.code() == docv1::HUMAN_LANGUAGE_LABEL_EN && lang0.code_raw() == "en" &&
                lang0.created_by() == "asr",
            "first item carries en language");
    require(lang1.code() == docv1::HUMAN_LANGUAGE_LABEL_ES && lang1.code_raw() == "es" &&
                lang1.created_by() == "asr",
            "second item carries es language");
    require(!document.body().meta().has_language(),
            "body.meta.language absent when multiple languages are present");
    require(document.source_meta().language().empty(),
            "source_meta.language empty when multiple languages are present");
}

void verify_per_channel_arrival_permutation_determinism() {
    auto ch_seg = [](uint32_t channel, uint32_t index, uint64_t start_ms, uint64_t end_ms,
                     const std::string& text, const std::string& speaker) {
        asrv1::TranscribeResponse event =
            segment_event(/*final=*/true, index, start_ms, end_ms, text, -0.25F);
        event.mutable_final_segment()->set_channel(channel);
        event.mutable_final_segment()->set_speaker_id(speaker);
        return event;
    };

    const asrv1::TranscribeResponse c0_first = ch_seg(0, 0, 0, 3000, " caller greeting", "C1");
    const asrv1::TranscribeResponse c1_first = ch_seg(1, 0, 0, 2800, " agent greeting", "C2");
    const asrv1::TranscribeResponse c0_second = ch_seg(0, 1, 5000, 8000, " caller question", "C1");
    const asrv1::TranscribeResponse c1_second = ch_seg(1, 1, 3500, 6000, " agent prompt", "C2");
    const asrv1::TranscribeResponse kf_early = keyframe_event(0);
    const asrv1::TranscribeResponse kf_late = keyframe_event(5000);

    auto run_permutation = [&](const std::vector<asrv1::TranscribeResponse>& events) {
        asr::doc::AsrDocumentFold fold("tiny.en", "1.2.3");
        fold.set_source("stereo.mp4", "video/mp4", 0xCAFEBABEULL);
        fold.consume(media_info_event());
        for (const asrv1::TranscribeResponse& ev : events) {
            fold.consume(ev);
        }
        fold.consume(complete_event());
        docv1::Document doc = fold.take();
        const std::vector<std::string> errors = asr::doc::document_integrity_errors(doc);
        require(errors.empty(),
                "per-channel permutation integrity: " + (errors.empty() ? "" : errors.front()));
        return doc;
    };

    const docv1::Document forward =
        run_permutation({kf_early, c0_first, c1_first, c1_second, kf_late, c0_second});
    const docv1::Document reversed =
        run_permutation({c0_second, kf_late, c1_second, c1_first, c0_first, kf_early});
    const docv1::Document interleaved =
        run_permutation({c1_first, c1_second, kf_late, c0_first, kf_early, c0_second});

    require(forward.media().speakers_size() == 2 && forward.media().speakers(0) == "C1" &&
                forward.media().speakers(1) == "C2",
            "speakers ordered deterministically as C1 then C2 on simultaneous t=0 speech");
    require(forward.texts_size() == 4 &&
                forward.texts(0).text().base().text() == "caller greeting" &&
                forward.texts(1).text().base().text() == "agent greeting" &&
                forward.texts(2).text().base().text() == "agent prompt" &&
                forward.texts(3).text().base().text() == "caller question",
            "texts arena canonicalized to (start_ms, channel) order");
    require(forward.body().children_size() == 6 &&
                forward.body().children(0).ref() == "#/pictures/0" &&
                forward.body().children(1).ref() == "#/texts/0" &&
                forward.body().children(2).ref() == "#/texts/1" &&
                forward.body().children(3).ref() == "#/texts/2" &&
                forward.body().children(4).ref() == "#/pictures/1" &&
                forward.body().children(5).ref() == "#/texts/3",
            "body.children references canonicalized arenas");

    auto deterministic_bytes = [](const google::protobuf::MessageLite& msg) {
        std::string out;
        {
            google::protobuf::io::StringOutputStream sink(&out);
            google::protobuf::io::CodedOutputStream coded(&sink);
            coded.SetSerializationDeterministic(true);
            require(msg.SerializeToCodedStream(&coded) && !coded.HadError(),
                    "deterministic protobuf serialization succeeded");
        }
        return out;
    };

    require(google::protobuf::util::MessageDifferencer::Equals(forward, reversed),
            "reversed arrival permutation equals forward Document");
    require(google::protobuf::util::MessageDifferencer::Equals(forward, interleaved),
            "interleaved arrival permutation equals forward Document");
    const std::string bytes_forward = deterministic_bytes(forward);
    require(bytes_forward == deterministic_bytes(reversed),
            "reversed arrival permutation yields byte-identical deterministic Document");
    require(bytes_forward == deterministic_bytes(interleaved),
            "interleaved arrival permutation yields byte-identical deterministic Document");
}

}  // namespace

int main() {
    try {
        verify_fold();
        verify_unprobed_media_facts_are_left_out();
        verify_media_measurements_are_not_made_up();
        verify_decode_warnings_fold_into_document_warnings();
        verify_word_provenance();
        verify_word_provenance_can_be_turned_off();
        verify_speaker_registry();
        verify_language_slots();
        verify_score_presence();
        verify_integrity_checker_catches_breakage();
        verify_time_ordered_body_with_early_keyframes();
        verify_suppressed_segment_skipped();
        verify_multilingual_fold();
        verify_per_channel_arrival_permutation_determinism();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("document-fold-test passed");
    return 0;
}
