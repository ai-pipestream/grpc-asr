#include "media/sniff.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <string>

namespace asr::media {

namespace {

constexpr uint8_t kAsfHeaderGuid[16] = {
    0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11,
    0xA6, 0xD9, 0x00, 0xAA, 0x00, 0x62, 0xCE, 0x6C,
};

// Cap on non-moov/non-mdat preamble boxes we wait to scan before falling back
// to the seekable memfd path.
constexpr uint64_t kMaxMp4PreambleScanBytes = 1024ULL * 1024ULL;

bool starts_with(const uint8_t* data, size_t size, const void* magic, size_t magic_size,
                 size_t offset = 0) {
    return size >= offset + magic_size && std::memcmp(data + offset, magic, magic_size) == 0;
}

uint32_t read_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

uint64_t read_be64(const uint8_t* p) {
    return (static_cast<uint64_t>(read_be32(p)) << 32) | static_cast<uint64_t>(read_be32(p + 4));
}

std::string lower_extension(std::string_view filename) {
    const size_t slash = filename.find_last_of("/\\");
    if (slash != std::string_view::npos) {
        filename.remove_prefix(slash + 1);
    }
    const size_t dot = filename.rfind('.');
    if (dot == std::string_view::npos || dot + 1 >= filename.size()) {
        return {};
    }
    std::string ext(filename.substr(dot));
    std::ranges::transform(
        ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

}  // namespace

bool decodes_in_process(MediaFamily family) {
    switch (family) {
        case MediaFamily::kWav:
        case MediaFamily::kMp3:
        case MediaFamily::kFlac:
            return true;
        default:
            return false;
    }
}

bool demuxes_with_ffmpeg(MediaFamily family) {
    switch (family) {
        case MediaFamily::kOgg:
        case MediaFamily::kMp4:
        case MediaFamily::kMkv:
        case MediaFamily::kAvi:
        case MediaFamily::kAsf:
        case MediaFamily::kAmr:
        case MediaFamily::kMpegPs:
        case MediaFamily::kMpegTs:
        case MediaFamily::kAac:
            return true;
        default:
            return false;
    }
}

MediaFamily sniff(const uint8_t* data, size_t size) {
    if (data == nullptr || size < 4) {
        return MediaFamily::kUnknown;
    }
    if (starts_with(data, size, "RIFF", 4)) {
        if (starts_with(data, size, "WAVE", 4, 8)) {
            return MediaFamily::kWav;
        }
        if (starts_with(data, size, "AVI ", 4, 8)) {
            return MediaFamily::kAvi;
        }
        return MediaFamily::kUnknown;
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
    if (data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3) {
        return MediaFamily::kMkv;
    }
    if (starts_with(data, size, kAsfHeaderGuid, sizeof(kAsfHeaderGuid))) {
        return MediaFamily::kAsf;
    }
    if (starts_with(data, size, "#!AMR-WB\n", 9) || starts_with(data, size, "#!AMR\n", 6)) {
        return MediaFamily::kAmr;
    }
    // MPEG-1 / MPEG-2 program stream pack header 00 00 01 BA.
    if (data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x01 && data[3] == 0xBA) {
        return MediaFamily::kMpegPs;
    }
    // MPEG-2 transport stream: three consecutive 188-byte packet sync bytes.
    if (size >= 377 && data[0] == 0x47 && data[188] == 0x47 && data[376] == 0x47) {
        return MediaFamily::kMpegTs;
    }
    // Raw ADTS AAC: 12-bit sync 0xFFF, ID bit 0 (MPEG-4) or 1 (MPEG-2),
    // layer 00, protection_absent 0 or 1 -> second byte 0xF0/0xF1/0xF8/0xF9.
    if (data[0] == 0xFF && (data[1] & 0xF6) == 0xF0) {
        return MediaFamily::kAac;
    }
    // ID3v2-tagged or bare MPEG audio (layers I/II/III, non-reserved version).
    if (starts_with(data, size, "ID3", 3)) {
        return MediaFamily::kMp3;
    }
    if (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0 && (data[1] & 0x18) != 0x08 &&
        (data[1] & 0x06) != 0) {
        return MediaFamily::kMp3;
    }
    return MediaFamily::kUnknown;
}

MediaFamily sniff_from_filename(std::string_view filename) {
    const std::string ext = lower_extension(filename);
    if (ext.empty()) {
        return MediaFamily::kUnknown;
    }
    if (ext == ".wav") {
        return MediaFamily::kWav;
    }
    if (ext == ".mp3") {
        return MediaFamily::kMp3;
    }
    if (ext == ".flac") {
        return MediaFamily::kFlac;
    }
    if (ext == ".ogg" || ext == ".oga" || ext == ".opus") {
        return MediaFamily::kOgg;
    }
    if (ext == ".mp4" || ext == ".m4a" || ext == ".m4v" || ext == ".mov") {
        return MediaFamily::kMp4;
    }
    if (ext == ".mkv" || ext == ".webm") {
        return MediaFamily::kMkv;
    }
    if (ext == ".avi") {
        return MediaFamily::kAvi;
    }
    if (ext == ".asf" || ext == ".wma" || ext == ".wmv") {
        return MediaFamily::kAsf;
    }
    if (ext == ".amr") {
        return MediaFamily::kAmr;
    }
    if (ext == ".mpg" || ext == ".mpeg") {
        return MediaFamily::kMpegPs;
    }
    if (ext == ".ts" || ext == ".m2ts") {
        return MediaFamily::kMpegTs;
    }
    if (ext == ".aac") {
        return MediaFamily::kAac;
    }
    return MediaFamily::kUnknown;
}

bool has_routed_media_extension(std::string_view filename) {
    return sniff_from_filename(filename) != MediaFamily::kUnknown;
}

StreamLayout sniff_stream_layout(MediaFamily family, const uint8_t* data, size_t size,
                                 bool stream_complete, size_t* needed_prefix) {
    if (needed_prefix != nullptr) {
        *needed_prefix = 0;
    }
    switch (family) {
        case MediaFamily::kOgg:
        case MediaFamily::kMkv:
        case MediaFamily::kAmr:
        case MediaFamily::kMpegPs:
        case MediaFamily::kMpegTs:
        case MediaFamily::kAac:
            return StreamLayout::kPipeable;
        case MediaFamily::kAvi:
        case MediaFamily::kAsf:
        case MediaFamily::kUnknown:
            return StreamLayout::kSeekableMemfd;
        case MediaFamily::kMp4:
            break;
        default:
            return StreamLayout::kSeekableMemfd;
    }

    if (data == nullptr) {
        return StreamLayout::kSeekableMemfd;
    }

    uint64_t offset = 0;
    while (offset <= kMaxMp4PreambleScanBytes) {
        if (size < offset + 8) {
            if (stream_complete) {
                return StreamLayout::kSeekableMemfd;
            }
            if (needed_prefix != nullptr) {
                *needed_prefix = static_cast<size_t>(offset + 16);
            }
            return StreamLayout::kNeedMoreBytes;
        }
        const uint32_t size32 = read_be32(data + offset);
        const std::string_view box_type(reinterpret_cast<const char*>(data + offset + 4), 4);
        uint64_t box_size = size32;
        if (size32 == 1) {
            if (size < offset + 16) {
                if (stream_complete) {
                    return StreamLayout::kSeekableMemfd;
                }
                if (needed_prefix != nullptr) {
                    *needed_prefix = static_cast<size_t>(offset + 16);
                }
                return StreamLayout::kNeedMoreBytes;
            }
            box_size = read_be64(data + offset + 8);
            if (box_size < 16) {
                return StreamLayout::kSeekableMemfd;
            }
        } else if (size32 == 0) {
            // Box extends to EOF: moov to EOF has no mdat after it anyway;
            // mdat to EOF means moov cannot precede it unless already seen.
            return box_type == "moov" ? StreamLayout::kPipeable : StreamLayout::kSeekableMemfd;
        } else if (size32 < 8) {
            return StreamLayout::kSeekableMemfd;
        }

        if (box_type == "moov") {
            return StreamLayout::kPipeable;
        }
        if (box_type == "mdat") {
            return StreamLayout::kSeekableMemfd;
        }
        if (box_size > kMaxMp4PreambleScanBytes || offset + box_size < offset) {
            return StreamLayout::kSeekableMemfd;
        }
        offset += box_size;
    }
    return StreamLayout::kSeekableMemfd;
}

ai::pipestream::asr::v1::ContainerFamily to_proto_container(MediaFamily family) {
    using ai::pipestream::asr::v1::ContainerFamily;
    switch (family) {
        case MediaFamily::kWav:
            return ContainerFamily::CONTAINER_FAMILY_WAV;
        case MediaFamily::kMp3:
            return ContainerFamily::CONTAINER_FAMILY_MP3;
        case MediaFamily::kFlac:
            return ContainerFamily::CONTAINER_FAMILY_FLAC;
        case MediaFamily::kOgg:
            return ContainerFamily::CONTAINER_FAMILY_OGG;
        case MediaFamily::kMp4:
            return ContainerFamily::CONTAINER_FAMILY_MP4;
        case MediaFamily::kMkv:
            return ContainerFamily::CONTAINER_FAMILY_MATROSKA;
        case MediaFamily::kAvi:
            return ContainerFamily::CONTAINER_FAMILY_AVI;
        case MediaFamily::kAsf:
            return ContainerFamily::CONTAINER_FAMILY_ASF;
        case MediaFamily::kAmr:
            return ContainerFamily::CONTAINER_FAMILY_AMR;
        case MediaFamily::kMpegPs:
            return ContainerFamily::CONTAINER_FAMILY_MPEG_PS;
        case MediaFamily::kMpegTs:
            return ContainerFamily::CONTAINER_FAMILY_MPEG_TS;
        case MediaFamily::kAac:
            return ContainerFamily::CONTAINER_FAMILY_ADTS_AAC;
        default:
            return ContainerFamily::CONTAINER_FAMILY_UNSPECIFIED;
    }
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
        case MediaFamily::kAvi:
            return "avi";
        case MediaFamily::kAsf:
            return "asf";
        case MediaFamily::kAmr:
            return "amr";
        case MediaFamily::kMpegPs:
            return "mpeg";
        case MediaFamily::kMpegTs:
            return "ts";
        case MediaFamily::kAac:
            return "aac";
        default:
            return "unknown";
    }
}

std::string_view family_mimetype(MediaFamily family) {
    switch (family) {
        case MediaFamily::kWav:
            return "audio/wav";
        case MediaFamily::kMp3:
            return "audio/mpeg";
        case MediaFamily::kFlac:
            return "audio/flac";
        case MediaFamily::kOgg:
            return "audio/ogg";
        case MediaFamily::kMp4:
            return "video/mp4";
        case MediaFamily::kMkv:
            return "video/x-matroska";
        case MediaFamily::kAvi:
            return "video/x-msvideo";
        case MediaFamily::kAsf:
            return "video/x-ms-asf";
        case MediaFamily::kAmr:
            return "audio/amr";
        case MediaFamily::kMpegPs:
            return "video/mpeg";
        case MediaFamily::kMpegTs:
            return "video/mp2t";
        case MediaFamily::kAac:
            return "audio/aac";
        default:
            return "application/octet-stream";
    }
}

std::string_view sniff_pipe_audio_codec(MediaFamily family, const uint8_t* data, size_t size) {
    if (data != nullptr) {
        if (family == MediaFamily::kOgg && starts_with(data, size, "OggS", 4) && size >= 27) {
            const size_t payload = 27 + static_cast<size_t>(data[26]);
            if (starts_with(data, size, "OpusHead", 8, payload)) {
                return "opus";
            }
            if (starts_with(data, size, "\x01vorbis", 7, payload)) {
                return "vorbis";
            }
            if (starts_with(data, size, "\x7F" "FLAC", 5, payload)) {
                return "flac";
            }
            if (starts_with(data, size, "Speex   ", 8, payload)) {
                return "speex";
            }
        }
        if (family == MediaFamily::kAmr) {
            if (starts_with(data, size, "#!AMR-WB\n", 9)) {
                return "amr_wb";
            }
            if (starts_with(data, size, "#!AMR\n", 6)) {
                return "amr_nb";
            }
        }
    }
    return family_name(family);
}

}  // namespace asr::media
