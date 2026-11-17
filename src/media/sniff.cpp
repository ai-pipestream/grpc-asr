#include "media/sniff.h"

#include <algorithm>
#include <array>
#include <bit>
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

namespace {

bool next_bmff_box(const uint8_t* data, size_t end, size_t* pos, std::string_view* type_out,
                   size_t* body_begin, size_t* body_end) {
    if (*pos + 8 > end) {
        return false;
    }
    const uint32_t size32 = read_be32(data + *pos);
    *type_out = std::string_view(reinterpret_cast<const char*>(data + *pos + 4), 4);
    uint64_t box_size = size32;
    size_t header_len = 8;
    if (size32 == 1) {
        if (*pos + 16 > end) {
            return false;
        }
        box_size = read_be64(data + *pos + 8);
        if (box_size < 16) {
            return false;
        }
        header_len = 16;
    } else if (size32 == 0) {
        box_size = end - *pos;
    } else if (size32 < 8) {
        return false;
    }
    if (box_size > end - *pos) {
        return false;
    }
    *body_begin = *pos + header_len;
    *body_end = *pos + static_cast<size_t>(box_size);
    *pos = *body_end;
    return true;
}

std::string_view esds_audio_codec(const uint8_t* data, size_t begin, size_t end) {
    // FullBox version/flags (4 B) precedes the ES_Descriptor tree; scan for
    // DecoderConfigDescriptor tag 0x04 followed by expandable length (1..4 B)
    // and objectTypeIndication.
    for (size_t i = begin + 4; i + 2 < end; ++i) {
        if (data[i] != 0x04) {
            continue;
        }
        size_t j = i + 1;
        size_t len_bytes = 0;
        uint32_t desc_len = 0;
        while (j < end && len_bytes < 4) {
            const uint8_t b = data[j++];
            desc_len = (desc_len << 7) | (b & 0x7FU);
            len_bytes++;
            if ((b & 0x80U) == 0) {
                break;
            }
        }
        if (j < end && desc_len >= 13 && j + desc_len <= end) {
            const uint8_t oti = data[j];
            if (oti == 0x69 || oti == 0x6B) {
                return "mp3";
            }
            if (oti == 0x40 || oti == 0x66 || oti == 0x67 || oti == 0x68) {
                return "aac";
            }
        }
    }
    return "aac";
}

std::string_view mp4a_sample_entry_codec(const uint8_t* data, size_t entry_begin,
                                         size_t entry_end) {
    // AudioSampleEntry: 8-byte box header + 28-byte fixed audio header = 36 B.
    if (entry_end - entry_begin <= 36) {
        return "aac";
    }
    size_t pos = entry_begin + 36;
    std::string_view type;
    size_t b_begin = 0;
    size_t b_end = 0;
    while (next_bmff_box(data, entry_end, &pos, &type, &b_begin, &b_end)) {
        if (type == "esds") {
            return esds_audio_codec(data, b_begin, b_end);
        }
        if (type == "wave") {
            size_t wpos = b_begin;
            std::string_view wtype;
            size_t wb_begin = 0;
            size_t wb_end = 0;
            while (next_bmff_box(data, b_end, &wpos, &wtype, &wb_begin, &wb_end)) {
                if (wtype == "esds") {
                    return esds_audio_codec(data, wb_begin, wb_end);
                }
            }
        }
    }
    return "aac";
}

std::string_view map_bmff_sample_entry(std::string_view fourcc, const uint8_t* data,
                                       size_t entry_begin, size_t entry_end) {
    if (fourcc == "mp4a") {
        return mp4a_sample_entry_codec(data, entry_begin, entry_end);
    }
    if (fourcc == "Opus") {
        return "opus";
    }
    if (fourcc == "fLaC") {
        return "flac";
    }
    if (fourcc == "alac") {
        return "alac";
    }
    if (fourcc == "ac-3") {
        return "ac3";
    }
    if (fourcc == "ec-3") {
        return "eac3";
    }
    if (fourcc == "samr") {
        return "amr_nb";
    }
    if (fourcc == "sawb") {
        return "amr_wb";
    }
    if (fourcc == ".mp3" || fourcc == std::string_view("ms\x00\x55", 4)) {
        return "mp3";
    }
    if (fourcc == "sowt" || fourcc == "lpcm") {
        return "pcm_s16le";
    }
    if (fourcc == "twos") {
        return "pcm_s16be";
    }
    return {};
}

std::string_view sniff_mp4_moov_codec(const uint8_t* data, size_t moov_begin, size_t moov_end) {
    size_t trak_pos = moov_begin;
    std::string_view trak_type;
    size_t trak_begin = 0;
    size_t trak_end = 0;
    while (next_bmff_box(data, moov_end, &trak_pos, &trak_type, &trak_begin, &trak_end)) {
        if (trak_type != "trak") {
            continue;
        }
        size_t mdia_pos = trak_begin;
        std::string_view mdia_type;
        size_t mdia_begin = 0;
        size_t mdia_end = 0;
        while (next_bmff_box(data, trak_end, &mdia_pos, &mdia_type, &mdia_begin, &mdia_end)) {
            if (mdia_type != "mdia") {
                continue;
            }
            bool is_sound = false;
            size_t minf_begin = 0;
            size_t minf_end = 0;
            size_t child_pos = mdia_begin;
            std::string_view child_type;
            size_t c_begin = 0;
            size_t c_end = 0;
            while (next_bmff_box(data, mdia_end, &child_pos, &child_type, &c_begin, &c_end)) {
                if (child_type == "hdlr" && c_end - c_begin >= 12) {
                    const std::string_view handler(
                        reinterpret_cast<const char*>(data + c_begin + 8), 4);
                    if (handler == "soun") {
                        is_sound = true;
                    }
                } else if (child_type == "minf") {
                    minf_begin = c_begin;
                    minf_end = c_end;
                }
            }
            if (!is_sound || minf_end <= minf_begin) {
                continue;
            }
            size_t minf_pos = minf_begin;
            std::string_view minf_type;
            size_t stbl_begin = 0;
            size_t stbl_end = 0;
            while (next_bmff_box(data, minf_end, &minf_pos, &minf_type, &stbl_begin, &stbl_end)) {
                if (minf_type != "stbl") {
                    continue;
                }
                size_t stbl_pos = stbl_begin;
                std::string_view stbl_type;
                size_t stsd_begin = 0;
                size_t stsd_end = 0;
                while (
                    next_bmff_box(data, stbl_end, &stbl_pos, &stbl_type, &stsd_begin, &stsd_end)) {
                    if (stbl_type != "stsd" || stsd_end - stsd_begin < 16) {
                        continue;
                    }
                    const size_t entry_begin = stsd_begin + 8;
                    const uint32_t entry_size = read_be32(data + entry_begin);
                    if (entry_size < 8) {
                        continue;
                    }
                    const size_t entry_end =
                        std::min(stsd_end, entry_begin + static_cast<size_t>(entry_size));
                    const std::string_view fourcc(
                        reinterpret_cast<const char*>(data + entry_begin + 4), 4);
                    const std::string_view mapped =
                        map_bmff_sample_entry(fourcc, data, entry_begin, entry_end);
                    if (!mapped.empty()) {
                        return mapped;
                    }
                }
            }
        }
    }
    return {};
}

std::string_view sniff_mp4_pipe_codec(const uint8_t* data, size_t size, bool stream_complete,
                                      size_t* needed_prefix) {
    uint64_t offset = 0;
    while (offset <= kMaxPipeCodecProbeBytes) {
        if (size < offset + 8) {
            if (!stream_complete && offset + 16 <= kMaxPipeCodecProbeBytes &&
                needed_prefix != nullptr) {
                *needed_prefix = static_cast<size_t>(offset + 16);
                return {};
            }
            break;
        }
        const uint32_t size32 = read_be32(data + offset);
        const std::string_view box_type(reinterpret_cast<const char*>(data + offset + 4), 4);
        uint64_t box_size = size32;
        size_t header_len = 8;
        if (size32 == 1) {
            if (size < offset + 16) {
                if (!stream_complete && offset + 16 <= kMaxPipeCodecProbeBytes &&
                    needed_prefix != nullptr) {
                    *needed_prefix = static_cast<size_t>(offset + 16);
                    return {};
                }
                break;
            }
            box_size = read_be64(data + offset + 8);
            if (box_size < 16) {
                break;
            }
            header_len = 16;
        } else if (size32 == 0) {
            box_size = size - offset;
        } else if (size32 < 8) {
            break;
        }

        if (box_type == "mdat") {
            break;
        }
        if (box_type == "moov") {
            const uint64_t moov_end_full = offset + box_size;
            if (moov_end_full > size && !stream_complete &&
                moov_end_full <= kMaxPipeCodecProbeBytes && needed_prefix != nullptr) {
                *needed_prefix = static_cast<size_t>(moov_end_full);
                return {};
            }
            const size_t moov_begin = static_cast<size_t>(offset) + header_len;
            const size_t moov_end =
                moov_end_full < size ? static_cast<size_t>(moov_end_full) : size;
            const std::string_view codec = sniff_mp4_moov_codec(data, moov_begin, moov_end);
            return codec.empty() ? std::string_view("mp4") : codec;
        }
        if (box_size > kMaxPipeCodecProbeBytes || offset + box_size < offset) {
            break;
        }
        offset += box_size;
    }
    return "mp4";
}

bool read_ebml_id(const uint8_t* data, size_t end, size_t* pos, uint32_t* id_out) {
    if (*pos >= end) {
        return false;
    }
    const uint8_t b0 = data[*pos];
    if (b0 == 0) {
        return false;
    }
    const size_t width = static_cast<size_t>(std::countl_zero(b0)) + 1;
    if (width > 4 || *pos + width > end) {
        return false;
    }
    uint32_t id = 0;
    for (size_t i = 0; i < width; ++i) {
        id = (id << 8) | data[*pos + i];
    }
    *pos += width;
    *id_out = id;
    return true;
}

bool read_ebml_size(const uint8_t* data, size_t end, size_t* pos, uint64_t* size_out,
                    bool* unknown_out) {
    if (*pos >= end) {
        return false;
    }
    const uint8_t b0 = data[*pos];
    if (b0 == 0) {
        return false;
    }
    const size_t width = static_cast<size_t>(std::countl_zero(b0)) + 1;
    if (*pos + width > end) {
        return false;
    }
    uint64_t val = b0 & ((1U << (8 - width)) - 1U);
    for (size_t i = 1; i < width; ++i) {
        val = (val << 8) | data[*pos + i];
    }
    const uint64_t all_ones =
        width == 8 ? 0x00FFFFFFFFFFFFFFULL : ((1ULL << (7 * width)) - 1ULL);
    *unknown_out = (val == all_ones);
    *size_out = val;
    *pos += width;
    return true;
}

std::string_view map_matroska_codec_id(std::string_view codec_id) {
    while (!codec_id.empty() && codec_id.back() == '\0') {
        codec_id.remove_suffix(1);
    }
    if (codec_id == "A_OPUS") {
        return "opus";
    }
    if (codec_id == "A_VORBIS") {
        return "vorbis";
    }
    if (codec_id.starts_with("A_AAC")) {
        return "aac";
    }
    if (codec_id == "A_FLAC") {
        return "flac";
    }
    if (codec_id == "A_MPEG/L3") {
        return "mp3";
    }
    if (codec_id == "A_MPEG/L2") {
        return "mp2";
    }
    if (codec_id == "A_MPEG/L1") {
        return "mp1";
    }
    if (codec_id == "A_AC3") {
        return "ac3";
    }
    if (codec_id == "A_EAC3") {
        return "eac3";
    }
    if (codec_id.starts_with("A_DTS")) {
        return "dts";
    }
    if (codec_id == "A_ALAC") {
        return "alac";
    }
    if (codec_id == "A_TRUEHD") {
        return "truehd";
    }
    if (codec_id == "A_PCM/INT/LIT") {
        return "pcm_s16le";
    }
    if (codec_id == "A_PCM/INT/BIG") {
        return "pcm_s16be";
    }
    if (codec_id == "A_PCM/FLOAT/IEEE") {
        return "pcm_f32le";
    }
    return {};
}

std::string_view parse_mkv_tracks(const uint8_t* data, size_t tracks_begin, size_t tracks_end) {
    size_t pos = tracks_begin;
    while (pos < tracks_end) {
        uint32_t entry_id = 0;
        uint64_t entry_size = 0;
        bool entry_unknown = false;
        if (!read_ebml_id(data, tracks_end, &pos, &entry_id) ||
            !read_ebml_size(data, tracks_end, &pos, &entry_size, &entry_unknown)) {
            break;
        }
        const size_t entry_end =
            entry_unknown || entry_size > tracks_end - pos
                ? tracks_end
                : pos + static_cast<size_t>(entry_size);
        if (entry_id == 0xAEU) {  // TrackEntry
            uint64_t track_type = 0;
            std::string_view codec_id;
            size_t field_pos = pos;
            while (field_pos < entry_end) {
                uint32_t field_id = 0;
                uint64_t field_size = 0;
                bool field_unknown = false;
                if (!read_ebml_id(data, entry_end, &field_pos, &field_id) ||
                    !read_ebml_size(data, entry_end, &field_pos, &field_size, &field_unknown) ||
                    field_unknown || field_size > entry_end - field_pos) {
                    break;
                }
                if (field_id == 0x83U && field_size >= 1 && field_size <= 8) {  // TrackType
                    uint64_t v = 0;
                    for (size_t i = 0; i < field_size; ++i) {
                        v = (v << 8) | data[field_pos + i];
                    }
                    track_type = v;
                } else if (field_id == 0x86U) {  // CodecID
                    codec_id = std::string_view(
                        reinterpret_cast<const char*>(data + field_pos),
                        static_cast<size_t>(field_size));
                }
                field_pos += static_cast<size_t>(field_size);
            }
            if (track_type == 2 && !codec_id.empty()) {
                const std::string_view mapped = map_matroska_codec_id(codec_id);
                if (!mapped.empty()) {
                    return mapped;
                }
            }
        }
        if (entry_unknown) {
            break;
        }
        pos = entry_end;
    }
    return {};
}

std::string_view sniff_mkv_pipe_codec(const uint8_t* data, size_t size, bool stream_complete,
                                      size_t* needed_prefix) {
    auto request_more = [&](size_t target) -> std::string_view {
        if (!stream_complete && size < kMaxPipeCodecProbeBytes && needed_prefix != nullptr) {
            const size_t clamped = std::min(std::max(target, size + 1024), kMaxPipeCodecProbeBytes);
            if (clamped > size) {
                *needed_prefix = clamped;
                return {};
            }
        }
        return "mkv";
    };

    size_t pos = 0;
    uint32_t top_id = 0;
    uint64_t top_size = 0;
    bool top_unknown = false;
    // Skip leading EBML header (0x1A45DFA3) to reach Segment (0x18538067).
    if (!read_ebml_id(data, size, &pos, &top_id) || top_id != 0x1A45DFA3U ||
        !read_ebml_size(data, size, &pos, &top_size, &top_unknown) || top_unknown) {
        return "mkv";
    }
    if (top_size > size - pos) {
        return request_more(pos + static_cast<size_t>(top_size) + 64);
    }
    pos += static_cast<size_t>(top_size);

    // Advance past any Void/CRC-32 elements before Segment.
    while (true) {
        if (pos + 12 > size && !stream_complete) {
            return request_more(pos + 1024);
        }
        if (!read_ebml_id(data, size, &pos, &top_id) ||
            !read_ebml_size(data, size, &pos, &top_size, &top_unknown)) {
            return request_more(size + 1024);
        }
        if (top_id == 0x18538067U) {  // Segment
            break;
        }
        if (top_unknown || top_size > kMaxPipeCodecProbeBytes) {
            return "mkv";
        }
        if (top_size > size - pos) {
            return request_more(pos + static_cast<size_t>(top_size) + 64);
        }
        pos += static_cast<size_t>(top_size);
    }

    const size_t seg_end =
        top_unknown || top_size > size - pos ? size : pos + static_cast<size_t>(top_size);

    while (pos < seg_end && pos < kMaxPipeCodecProbeBytes) {
        if (pos + 12 > size && !stream_complete) {
            return request_more(pos + 2048);
        }
        uint32_t child_id = 0;
        uint64_t child_size = 0;
        bool child_unknown = false;
        if (!read_ebml_id(data, seg_end, &pos, &child_id) ||
            !read_ebml_size(data, seg_end, &pos, &child_size, &child_unknown)) {
            return request_more(size + 2048);
        }
        if (child_id == 0x1F43B675U) {  // Cluster: media payload started; never wait past here.
            break;
        }
        if (child_id == 0x1654AE6BU) {  // Tracks
            if (!child_unknown && pos + child_size > size) {
                if (!stream_complete && pos + child_size <= kMaxPipeCodecProbeBytes &&
                    needed_prefix != nullptr) {
                    *needed_prefix = pos + static_cast<size_t>(child_size);
                    return {};
                }
            }
            const size_t tracks_end =
                child_unknown || child_size > seg_end - pos
                    ? seg_end
                    : pos + static_cast<size_t>(child_size);
            const std::string_view codec = parse_mkv_tracks(data, pos, tracks_end);
            return codec.empty() ? std::string_view("mkv") : codec;
        }
        if (child_unknown || child_size > kMaxPipeCodecProbeBytes) {
            break;
        }
        if (child_size > seg_end - pos) {
            return request_more(pos + static_cast<size_t>(child_size) + 64);
        }
        pos += static_cast<size_t>(child_size);
    }
    return "mkv";
}

}  // namespace

std::string_view sniff_pipe_audio_codec(MediaFamily family, const uint8_t* data, size_t size,
                                        bool stream_complete, size_t* needed_prefix) {
    if (needed_prefix != nullptr) {
        *needed_prefix = 0;
    }
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
        if (family == MediaFamily::kMp4) {
            return sniff_mp4_pipe_codec(data, size, stream_complete, needed_prefix);
        }
        if (family == MediaFamily::kMkv) {
            return sniff_mkv_pipe_codec(data, size, stream_complete, needed_prefix);
        }
    }
    return family_name(family);
}

}  // namespace asr::media
