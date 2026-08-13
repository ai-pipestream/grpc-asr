#include "media/sniff.h"

#include <cstring>

namespace asr::media {

namespace {

bool starts_with(const uint8_t* data, size_t size, const char* magic, size_t magic_size,
                 size_t offset = 0) {
    return size >= offset + magic_size && std::memcmp(data + offset, magic, magic_size) == 0;
}

}  // namespace

bool is_audio_family(MediaFamily family) {
    switch (family) {
        case MediaFamily::kWav:
        case MediaFamily::kMp3:
        case MediaFamily::kFlac:
        case MediaFamily::kOgg:
            return true;
        default:
            return false;
    }
}

bool is_video_family(MediaFamily family) {
    return family == MediaFamily::kMp4 || family == MediaFamily::kMkv;
}

MediaFamily sniff(const uint8_t* data, size_t size) {
    if (data == nullptr || size < 4) {
        return MediaFamily::kUnknown;
    }
    if (starts_with(data, size, "RIFF", 4) && starts_with(data, size, "WAVE", 4, 8)) {
        return MediaFamily::kWav;
    }
    if (starts_with(data, size, "fLaC", 4)) {
        return MediaFamily::kFlac;
    }
    if (starts_with(data, size, "OggS", 4)) {
        return MediaFamily::kOgg;
    }
    if (starts_with(data, size, "ftyp", 4, 4)) {
        return MediaFamily::kMp4;
    }
    // EBML header: mkv and webm share it; ffmpeg tells them apart later.
    if (size >= 4 && data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3) {
        return MediaFamily::kMkv;
    }
    // ID3v2-tagged or bare MPEG audio. A bare frame starts with an 11-bit
    // sync run; check the first two bytes only, the decoder validates the
    // rest.
    if (starts_with(data, size, "ID3", 3)) {
        return MediaFamily::kMp3;
    }
    if (size >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
        return MediaFamily::kMp3;
    }
    return MediaFamily::kUnknown;
}

std::string_view family_name(MediaFamily family) {
    switch (family) {
        case MediaFamily::kWav:
            return "wav";
        case MediaFamily::kMp3:
            return "mp3";
        case MediaFamily::kFlac:
            return "flac";
        case MediaFamily::kOgg:
            return "ogg";
        case MediaFamily::kMp4:
            return "mp4";
        case MediaFamily::kMkv:
            return "mkv";
        default:
            return "unknown";
    }
}

}  // namespace asr::media
