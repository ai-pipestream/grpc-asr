#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ai/pipestream/asr/v1/asr_service.pb.h"
#include "ai/pipestream/document/v1/document.pb.h"

namespace asr::doc {

// Per-fold knobs, resolved from server config.
struct FoldOptions {
    // Whether text items locate every word or only the whole segment.
    // On by default. The cost is one ProvenanceItem per word, roughly 150
    // per transcribed minute: a ten-minute clip adds about 1500 entries,
    // an eight-hour recording about 72000, and the Document is one bounded
    // gRPC message. Operators serving very long media turn it off; the
    // typed stream still carries every word either way. Words exist at all
    // only when the client set TranscribeOptions.word_timestamps.
    bool word_provenance = true;
};

// AsrDocumentFold folds a Transcribe response stream into one
// ai.pipestream.document.v1.Document — the collector-side projection the
// fleet's scatter-gather coordinator merges additively (the canonical fold
// lives next to the collector). It consumes the same TranscribeResponse
// messages the server writes, in write order, with O(1) work per event
// plus appends, and is structurally valid at any point mid-stream.
//
// The typed event stream is the lossless wire; this fold is the lossy one.
//
// Time is provenance here, not attribution: every text item and every
// picture carries `prov` entries whose `time` is the media span they came
// from — one for the segment, then one per word when the decoder aligned
// words and word provenance is on. Each word entry also carries the
// charspan (Unicode code points) of that word inside the item's own text,
// so a consumer can highlight in place. The legacy TrackSource stays on
// every item alongside the collector attribution
// (CollectorSource{collector "asr", model, version, confidence from the
// segment's avg logprob}) for consumers that already read it, and the raw
// unrescaled score rides the same CollectorSource as raw_score with
// raw_score_kind "avg_logprob", so rescoring is still possible without
// inverting the rescale. Speakers, when the decoder diarizes, are
// stream-local labels on the time spans, registered once in
// Document.media.speakers.
//
// Keyframes become picture items whose ImageRef.uri is the pointer
// "keyframe:<timestamp_ms>" back into the typed stream — PNG bytes are
// never embedded, so the Document stays one bounded gRPC message. Partial
// segments are skipped (their finals supersede them). Media has no pages,
// so page_no stays 0 and no bbox is invented.
//
// What still stays on the typed stream: per-word probabilities (the
// schema has no per-word confidence slot), the segment index, and the
// trailer's segment and keyframe counts.
class AsrDocumentFold {
  public:
    // model and version go into every item's CollectorSource: model is the
    // whisper model id serving the stream, version this server's build.
    AsrDocumentFold(std::string model, std::string version, FoldOptions options = {});

    // Names the source media, which the event stream does not carry:
    // Document.name plus the DocumentOrigin the coordinator merges into
    // its base document. May be called at any point before take().
    void set_source(std::string filename, std::string mimetype, uint64_t binary_hash);

    // Consumes one server-to-client event; document events and partial
    // segments are ignored.
    void consume(const ai::pipestream::asr::v1::TranscribeResponse& event);

    // True once the TranscriptComplete trailer has been folded.
    bool finished() const { return finished_; }

    const ai::pipestream::document::v1::Document& document() const { return document_; }

    // Moves the finished Document out; the fold must not be fed afterwards.
    ai::pipestream::document::v1::Document take() { return std::move(document_); }

  private:
    void on_media_info(const ai::pipestream::asr::v1::MediaInfo& info);
    void on_final_segment(const ai::pipestream::asr::v1::Segment& segment);
    void on_keyframe(const ai::pipestream::asr::v1::Keyframe& keyframe);
    void on_complete(const ai::pipestream::asr::v1::TranscriptComplete& complete);

    // Appends one ProvenanceItem per word of the segment, each located by
    // its media time span and by its charspan inside `text`.
    void add_word_provenance(ai::pipestream::document::v1::TextItemBase* base,
                             const ai::pipestream::asr::v1::Segment& segment,
                             const std::string& text);

    // Adds a speaker label to Document.media in first-appearance order,
    // ignoring one already registered.
    void register_speaker(const std::string& speaker);

    // Appends the TrackSource + CollectorSource pair every item carries.
    // An absent avg_logprob means the decoder scored no token, so the item
    // claims neither a confidence nor a raw score; a present 0 is a real
    // score and lands in both.
    void stamp_sources(
        google::protobuf::RepeatedPtrField<ai::pipestream::document::v1::SourceType>* source,
        double start_seconds, double end_seconds, std::optional<float> avg_logprob);

    ai::pipestream::document::v1::Document document_;
    std::string model_;
    std::string version_;
    FoldOptions options_;
    bool finished_ = false;
};

// Structural self-check over the arenas this fold populates (body,
// furniture, groups, texts, pictures, tables): empty or duplicate
// self_refs, child refs that do not resolve, parent refs that do not
// resolve, and parents that do not list the item among their children.
// Empty means structurally sound; tests assert that after every fold.
std::vector<std::string> document_integrity_errors(
    const ai::pipestream::document::v1::Document& document);

}  // namespace asr::doc
