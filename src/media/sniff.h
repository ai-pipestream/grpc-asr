#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace asr::media {

// Container family detected from magic bytes. Drives the decode path:
// wav, mp3 and flac decode in-process, everything else demuxes via
// ffmpeg.
enum class MediaFamily {
    kUnknown,
    kWav,
    kMp3,
    kFlac,
    kOgg,
    kMp4,   // mp4 / mov / m4a (ISO BMFF)
    kMkv,   // mkv / webm (EBML)
};

// True for families the in-process decoder (miniaudio) handles: wav, mp3,
// flac.
bool decodes_in_process(MediaFamily family);

// True for families demuxed and decoded through an ffmpeg child: the
// containers that may carry video (mp4, mkv), and ogg, whose Vorbis and
// Opus audio miniaudio cannot decode here.
bool demuxes_with_ffmpeg(MediaFamily family);

// Sniffs the container family from the first bytes of the media.
// Needs at most 16 bytes; shorter inputs return kUnknown.
MediaFamily sniff(const uint8_t* data, size_t size);

// Human-readable family name for logs and errors, e.g. "wav", "mp4".
// Doubles as the filename extension when a name has to be derived.
std::string_view family_name(MediaFamily family);

// MIME type for the sniffed container, e.g. "audio/wav", "video/mp4".
// The container families that can hold either audio or video (mp4/m4a,
// mkv/webm) report the video type: the magic bytes cannot tell them
// apart, and only the demuxer knows. Unknown containers report
// "application/octet-stream".
std::string_view family_mimetype(MediaFamily family);

}  // namespace asr::media
