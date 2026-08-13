#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace asr::media {

// Container family detected from magic bytes. Drives the decode path:
// audio families decode in-process, video families demux via ffmpeg.
enum class MediaFamily {
    kUnknown,
    kWav,
    kMp3,
    kFlac,
    kOgg,
    kMp4,   // mp4 / mov / m4a (ISO BMFF)
    kMkv,   // mkv / webm (EBML)
};

// True for families handled by the in-process audio decoder.
bool is_audio_family(MediaFamily family);

// True for container families that may carry a video stream and are
// demuxed through ffmpeg.
bool is_video_family(MediaFamily family);

// Sniffs the container family from the first bytes of the media.
// Needs at most 16 bytes; shorter inputs return kUnknown.
MediaFamily sniff(const uint8_t* data, size_t size);

// Human-readable family name for logs and errors, e.g. "wav", "mp4".
std::string_view family_name(MediaFamily family);

}  // namespace asr::media
