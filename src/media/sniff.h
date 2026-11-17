#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "ai/pipestream/asr/v1/asr_service.pb.h"

namespace asr::media {

// Prefix size sufficient for every container signature, including three 188 B
// MPEG-TS packets (sync 0x47 at offsets 0, 188 and 376).
inline constexpr size_t kSniffPrefixBytes = 400;

// Upper bound on header bytes inspected to extract an audio codec name before
// the first media packet on the live pipe path.
inline constexpr size_t kMaxPipeCodecProbeBytes = 64 * 1024;

// Container family detected from magic bytes or routed filename extension.
// Drives the decode path: wav, mp3 and flac decode in-process when recognized
// by magic; everything else demuxes via ffmpeg.
enum class MediaFamily {
    kUnknown,
    kWav,
    kMp3,
    kFlac,
    kOgg,
    kMp4,     // mp4 / mov / m4a / m4v (ISO BMFF)
    kMkv,     // mkv / webm (EBML)
    kAvi,     // RIFF AVI
    kAsf,     // ASF / WMA / WMV
    kAmr,     // AMR-NB / AMR-WB
    kMpegPs,  // MPEG-1 / MPEG-2 program stream (.mpg / .mpeg)
    kMpegTs,  // MPEG-2 transport stream (.ts / .m2ts)
    kAac,     // Raw ADTS AAC (.aac)
};

// Streaming vs seekable container layout for the ffmpeg path.
enum class StreamLayout {
    // Prefix does not yet reach either `moov` or `mdat` in an ISO BMFF stream.
    kNeedMoreBytes,
    // Demuxable from non-seekable `pipe:0` while the upload is still in flight
    // (ogg, mkv/webm, fast-start mp4 with `moov` before `mdat`, mpegps, mpegts,
    // amr, raw adts aac).
    kPipeable,
    // Requires a seekable sealed memfd (trailing-`moov` mp4, avi, asf, or
    // unrecognized extension-fallback containers).
    kSeekableMemfd,
};

// True for families the in-process decoder (miniaudio) handles: wav, mp3, flac.
bool decodes_in_process(MediaFamily family);

// True for families demuxed and decoded through an ffmpeg child.
bool demuxes_with_ffmpeg(MediaFamily family);

// Sniffs the container family from the leading bytes of the media.
MediaFamily sniff(const uint8_t* data, size_t size);

// Fallback mapping from a filename's extension for the extensions gRParse
// routes to the ASR collector. Returns kUnknown when the extension is absent
// or not a routed media extension.
MediaFamily sniff_from_filename(std::string_view filename);

// True when `filename` ends (case-insensitively) in one of the extensions
// gRParse routes to ASR.
bool has_routed_media_extension(std::string_view filename);

// Classifies whether an ffmpeg-demuxed container can stream over `pipe:0`
// while uploading or must wait for a seekable memfd. When `kNeedMoreBytes` is
// returned, `*needed_prefix` (if non-null) is set to the next ISO BMFF box
// header offset to wait for.
StreamLayout sniff_stream_layout(MediaFamily family, const uint8_t* data, size_t size,
                                 bool stream_complete, size_t* needed_prefix = nullptr);

// Maps the internal MediaFamily to the wire ContainerFamily enum.
ai::pipestream::asr::v1::ContainerFamily to_proto_container(MediaFamily family);

// Human-readable family name for logs and errors, e.g. "wav", "mp4".
// Doubles as the filename extension when a name has to be derived.
std::string_view family_name(MediaFamily family);

// MIME type for the sniffed container, e.g. "audio/wav", "video/mp4".
std::string_view family_mimetype(MediaFamily family);

// Best-effort audio codec name derived from the leading container header bytes
// when ffprobe is skipped on the live pipe/remote path (Ogg BOS Opus/Vorbis/
// FLAC/Speex, AMR-NB/WB magic, Matroska/WebM EBML Tracks CodecID, and ISO BMFF
// fast-start moov/trak/mdia/minf/stbl/stsd sample entry fourcc, falling back to
// family_name(family)). When `stream_complete` is false and a bounded header
// box (`moov` or EBML `Tracks` prior to `Cluster`) extends past `size`, returns
// an empty string_view and sets `*needed_prefix` (<= kMaxPipeCodecProbeBytes)
// to the header offset to wait for.
std::string_view sniff_pipe_audio_codec(MediaFamily family, const uint8_t* data, size_t size,
                                        bool stream_complete = true,
                                        size_t* needed_prefix = nullptr);

}  // namespace asr::media
