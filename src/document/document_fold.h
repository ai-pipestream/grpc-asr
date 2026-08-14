#pragma once

#include <string>
#include <vector>

#include "ai/pipestream/asr/v1/asr_service.pb.h"
#include "ai/pipestream/document/v1/document.pb.h"

namespace asr::doc {

// AsrDocumentFold folds a Transcribe response stream into one
// ai.pipestream.document.v1.Document — the collector-side projection the
// fleet's scatter-gather coordinator merges additively (the DoclingMapper
// precedent: the canonical fold lives next to the collector). It consumes
// the same TranscribeResponse messages the server writes, in write order,
// with O(1) work per event plus appends, and is structurally valid at any
// point mid-stream.
//
// The typed event stream is the lossless wire; this fold is the lossy one.
// Final segments become text items whose sources carry both the media time
// range (TrackSource, seconds) and the collector attribution
// (CollectorSource{collector "asr", model, version, confidence from the
// segment's avg logprob}). Keyframes become picture items whose
// ImageRef.uri is the pointer "keyframe:<timestamp_ms>" back into the typed
// stream — PNG bytes are never embedded, so the Document stays one bounded
// gRPC message. Partial segments are skipped (their finals supersede them),
// and per-word timings stay on the typed stream only. Media has no pages,
// so no item carries provenance; timing lives in the TrackSource.
class AsrDocumentFold {
  public:
    // model and version go into every item's CollectorSource: model is the
    // whisper model id serving the stream, version this server's build.
    AsrDocumentFold(std::string model, std::string version);

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

    // Appends the TrackSource + CollectorSource pair every item carries.
    void stamp_sources(
        google::protobuf::RepeatedPtrField<ai::pipestream::document::v1::SourceType>* source,
        double start_seconds, double end_seconds, float avg_logprob);

    ai::pipestream::document::v1::Document document_;
    std::string model_;
    std::string version_;
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
