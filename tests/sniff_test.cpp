// Container sniffing, routed filename fallback, and stream layout detection
// over hand-authored magic bytes.

#include "media/sniff.h"

#include <initializer_list>
#include <print>
#include <string>

#include "fixture.h"

using asr::media::MediaFamily;
using asr::media::StreamLayout;
using asr::media::sniff;
using asr::media::sniff_from_filename;
using asr::media::sniff_stream_layout;
namespace asrv1 = ai::pipestream::asr::v1;

namespace {

MediaFamily sniff_str(const std::string& bytes) {
    return sniff(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
}

std::string box(uint32_t size, std::string_view type, size_t payload_pad = 0) {
    std::string out;
    out.push_back(static_cast<char>((size >> 24) & 0xFF));
    out.push_back(static_cast<char>((size >> 16) & 0xFF));
    out.push_back(static_cast<char>((size >> 8) & 0xFF));
    out.push_back(static_cast<char>(size & 0xFF));
    out.append(type);
    out.append(payload_pad, '\0');
    return out;
}

void verify_families() {
    require(sniff_str(make_wav(0.01, 0)) == MediaFamily::kWav, "wav sniffs as wav");
    require(sniff_str("RIFFxxxxAVI ") == MediaFamily::kAvi, "RIFF AVI sniffs as avi");
    require(sniff_str(make_truncated_mp3()) == MediaFamily::kMp3, "ID3 sniffs as mp3");
    require(sniff_str(std::string("\xFF\xFB\x90\x00", 4)) == MediaFamily::kMp3,
            "bare MPEG sync sniffs as mp3");
    require(sniff_str(std::string("\xFF\xF3\x90\x00", 4)) == MediaFamily::kMp3,
            "an MPEG-2 layer III frame sniffs as mp3");
    require(sniff_str(std::string("\xFF\xE3\x90\x00", 4)) == MediaFamily::kMp3,
            "an MPEG-2.5 layer III frame sniffs as mp3");
    require(sniff_str("fLaC\x00\x00\x00\x22") == MediaFamily::kFlac, "flac magic");
    require(sniff_str("OggS\x00\x02") == MediaFamily::kOgg, "ogg magic");
    require(sniff_str(std::string("\x00\x00\x00\x20", 4) + "ftypisom") == MediaFamily::kMp4,
            "ftyp at offset 4 sniffs as mp4");
    require(sniff_str(std::string("\x1A\x45\xDF\xA3", 4) + "junk") == MediaFamily::kMkv,
            "EBML sniffs as mkv");

    const std::string asf_guid(
        "\x30\x26\xB2\x75\x8E\x66\xCF\x11\xA6\xD9\x00\xAA\x00\x62\xCE\x6C", 16);
    require(sniff_str(asf_guid) == MediaFamily::kAsf, "ASF header GUID sniffs as asf");
    require(sniff_str("#!AMR\n\x3c\x00") == MediaFamily::kAmr, "AMR-NB magic");
    require(sniff_str("#!AMR-WB\n\x04\x00") == MediaFamily::kAmr, "AMR-WB magic");
    require(sniff_str(std::string("\x00\x00\x01\xBA\x44\x00", 6)) == MediaFamily::kMpegPs,
            "MPEG-PS pack header sniffs as mpegps");

    std::string ts(asr::media::kSniffPrefixBytes, '\0');
    ts[0] = 0x47;
    ts[188] = 0x47;
    ts[376] = 0x47;
    require(sniff_str(ts) == MediaFamily::kMpegTs, "MPEG-TS 188B sync run sniffs as mpegts");

    require(sniff_str(std::string("\xFF\xF1\x50\x80", 4)) == MediaFamily::kAac,
            "MPEG-4 ADTS AAC sniffs as aac");
    require(sniff_str(std::string("\xFF\xF9\x50\x80", 4)) == MediaFamily::kAac,
            "MPEG-2 ADTS AAC sniffs as aac");
}

void verify_rejects() {
    require(sniff_str("plain text, not media at all") == MediaFamily::kUnknown,
            "text is unknown");
    require(sniff_str("") == MediaFamily::kUnknown, "empty is unknown");
    require(sniff_str("RI") == MediaFamily::kUnknown, "short prefix is unknown");
    require(sniff_str("RIFFxxxxWEBP") == MediaFamily::kUnknown,
            "RIFF WEBP image is unknown");
    require(sniff_str(std::string("\xFF\xEB\x90\x00", 4)) == MediaFamily::kUnknown,
            "a frame with the reserved MPEG version is not mp3 or aac");
    // Single 'G' (0x47) text file is not MPEG-TS without sync at 188 and 376.
    require(sniff_str(std::string(400, 'A').replace(0, 1, "G")) == MediaFamily::kUnknown,
            "leading G without 188/376 sync is not MPEG-TS");
}

void verify_filename_fallback() {
    // Every extension gRParse routes to ASR must map to a known family.
    const std::pair<const char*, MediaFamily> routed[] = {
        {"clip.mp3", MediaFamily::kMp3},   {"clip.WAV", MediaFamily::kWav},
        {"clip.m4a", MediaFamily::kMp4},   {"clip.aac", MediaFamily::kAac},
        {"clip.flac", MediaFamily::kFlac}, {"clip.ogg", MediaFamily::kOgg},
        {"clip.oga", MediaFamily::kOgg},   {"clip.opus", MediaFamily::kOgg},
        {"clip.wma", MediaFamily::kAsf},   {"clip.amr", MediaFamily::kAmr},
        {"clip.mp4", MediaFamily::kMp4},   {"clip.m4v", MediaFamily::kMp4},
        {"clip.mkv", MediaFamily::kMkv},   {"clip.webm", MediaFamily::kMkv},
        {"clip.mov", MediaFamily::kMp4},   {"clip.avi", MediaFamily::kAvi},
        {"clip.mpg", MediaFamily::kMpegPs},{"clip.mpeg", MediaFamily::kMpegPs},
        {"clip.wmv", MediaFamily::kAsf},   {"/dir/clip.ts", MediaFamily::kMpegTs},
    };
    for (const auto& [name, expected] : routed) {
        require(asr::media::has_routed_media_extension(name),
                std::string(name) + " is recognized as routed");
        require(sniff_from_filename(name) == expected,
                std::string(name) + " maps to expected family");
    }
    require(!asr::media::has_routed_media_extension("notes.txt") &&
                !asr::media::has_routed_media_extension("archive.pdf") &&
                !asr::media::has_routed_media_extension(""),
            "non-media extensions are not routed");
}

void verify_stream_layouts() {
    for (MediaFamily f : {MediaFamily::kOgg, MediaFamily::kMkv, MediaFamily::kAmr,
                          MediaFamily::kMpegPs, MediaFamily::kMpegTs, MediaFamily::kAac}) {
        require(sniff_stream_layout(f, nullptr, 0, false) == StreamLayout::kPipeable,
                std::string(asr::media::family_name(f)) + " is pipeable");
    }
    for (MediaFamily f : {MediaFamily::kAvi, MediaFamily::kAsf, MediaFamily::kUnknown}) {
        require(sniff_stream_layout(f, nullptr, 0, false) == StreamLayout::kSeekableMemfd,
                std::string(asr::media::family_name(f)) + " uses seekable memfd");
    }

    // Fast-start MP4: ftyp (16B) -> free (8B) -> moov (64B) -> mdat.
    const std::string faststart = box(16, "ftyp", 8) + box(8, "free") + box(64, "moov", 8);
    require(sniff_stream_layout(MediaFamily::kMp4,
                                reinterpret_cast<const uint8_t*>(faststart.data()),
                                faststart.size(), false) == StreamLayout::kPipeable,
            "fast-start mp4 with moov before mdat is pipeable");

    // Trailing-moov MP4: ftyp (16B) -> mdat (1024B).
    const std::string trailing = box(16, "ftyp", 8) + box(1024, "mdat", 8);
    require(sniff_stream_layout(MediaFamily::kMp4,
                                reinterpret_cast<const uint8_t*>(trailing.data()), trailing.size(),
                                false) == StreamLayout::kSeekableMemfd,
            "trailing-moov mp4 with mdat first uses seekable memfd");

    // Incomplete prefix (only ftyp arrived so far while upload is open).
    const std::string partial = box(16, "ftyp", 8);
    size_t needed = 0;
    require(sniff_stream_layout(MediaFamily::kMp4,
                                reinterpret_cast<const uint8_t*>(partial.data()), partial.size(),
                                /*stream_complete=*/false, &needed) == StreamLayout::kNeedMoreBytes,
            "ftyp-only open prefix asks for more bytes");
    require(needed > partial.size(), "needed_prefix advances past ftyp");
    require(sniff_stream_layout(MediaFamily::kMp4,
                                reinterpret_cast<const uint8_t*>(partial.data()), partial.size(),
                                /*stream_complete=*/true) == StreamLayout::kSeekableMemfd,
            "truncated complete mp4 falls back to memfd");
}

void verify_family_kinds_and_proto() {
    using asr::media::decodes_in_process;
    using asr::media::demuxes_with_ffmpeg;
    using asr::media::to_proto_container;

    for (MediaFamily family : {MediaFamily::kWav, MediaFamily::kMp3, MediaFamily::kFlac}) {
        require(decodes_in_process(family) && !demuxes_with_ffmpeg(family),
                std::string(asr::media::family_name(family)) + " decodes in process");
    }
    for (MediaFamily family :
         {MediaFamily::kOgg, MediaFamily::kMp4, MediaFamily::kMkv, MediaFamily::kAvi,
          MediaFamily::kAsf, MediaFamily::kAmr, MediaFamily::kMpegPs, MediaFamily::kMpegTs,
          MediaFamily::kAac}) {
        require(demuxes_with_ffmpeg(family) && !decodes_in_process(family),
                std::string(asr::media::family_name(family)) + " goes through ffmpeg");
        require(to_proto_container(family) != asrv1::CONTAINER_FAMILY_UNSPECIFIED,
                std::string(asr::media::family_name(family)) + " maps to proto ContainerFamily");
        require(asr::media::family_mimetype(family) != "application/octet-stream",
                std::string(asr::media::family_name(family)) + " has specific mimetype");
    }
    require(!decodes_in_process(MediaFamily::kUnknown) &&
                !demuxes_with_ffmpeg(MediaFamily::kUnknown) &&
                to_proto_container(MediaFamily::kUnknown) == asrv1::CONTAINER_FAMILY_UNSPECIFIED &&
                asr::media::family_mimetype(MediaFamily::kUnknown) == "application/octet-stream",
            "unknown claims nothing");

    std::string ogg_opus(28, '\0');
    ogg_opus.replace(0, 4, "OggS");
    ogg_opus[26] = 1;
    ogg_opus += "OpusHead";
    require(asr::media::sniff_pipe_audio_codec(
                MediaFamily::kOgg, reinterpret_cast<const uint8_t*>(ogg_opus.data()),
                ogg_opus.size()) == "opus",
            "Ogg BOS OpusHead sniffs as opus");

    std::string ogg_vorbis(28, '\0');
    ogg_vorbis.replace(0, 4, "OggS");
    ogg_vorbis[26] = 1;
    ogg_vorbis.append("\x01vorbis", 7);
    require(asr::media::sniff_pipe_audio_codec(
                MediaFamily::kOgg, reinterpret_cast<const uint8_t*>(ogg_vorbis.data()),
                ogg_vorbis.size()) == "vorbis",
            "Ogg BOS vorbis sniffs as vorbis");

    require(asr::media::sniff_pipe_audio_codec(
                MediaFamily::kAmr, reinterpret_cast<const uint8_t*>("#!AMR\n\x3c"), 7) ==
                "amr_nb",
            "AMR-NB header sniffs as amr_nb");

    // Fast-start MP4 moov/trak/mdia/(hdlr 'soun' + minf/stbl/stsd 'mp4a' and 'Opus').
    auto wrap_box = [](std::string_view type, const std::string& body) {
        const auto sz = static_cast<uint32_t>(8 + body.size());
        std::string out;
        out.push_back(static_cast<char>((sz >> 24) & 0xFF));
        out.push_back(static_cast<char>((sz >> 16) & 0xFF));
        out.push_back(static_cast<char>((sz >> 8) & 0xFF));
        out.push_back(static_cast<char>(sz & 0xFF));
        out.append(type);
        out.append(body);
        return out;
    };
    auto make_mp4_header = [&](std::string_view sample_fourcc) {
        std::string hdlr_body(8, '\0');
        hdlr_body += "soun";
        hdlr_body.append(12, '\0');
        std::string stsd_body(8, '\0');
        stsd_body[7] = 1;  // entry_count = 1
        stsd_body += wrap_box(sample_fourcc, std::string(28, '\0'));
        const std::string mdia =
            wrap_box("hdlr", hdlr_body) +
            wrap_box("minf", wrap_box("stbl", wrap_box("stsd", stsd_body)));
        return box(16, "ftyp", 8) + wrap_box("moov", wrap_box("trak", wrap_box("mdia", mdia)));
    };
    const std::string mp4_aac = make_mp4_header("mp4a");
    require(asr::media::sniff_pipe_audio_codec(
                MediaFamily::kMp4, reinterpret_cast<const uint8_t*>(mp4_aac.data()),
                mp4_aac.size()) == "aac",
            "fast-start MP4 moov/stsd mp4a sniffs as aac");
    const std::string mp4_opus = make_mp4_header("Opus");
    require(asr::media::sniff_pipe_audio_codec(
                MediaFamily::kMp4, reinterpret_cast<const uint8_t*>(mp4_opus.data()),
                mp4_opus.size()) == "opus",
            "fast-start MP4 moov/stsd Opus sniffs as opus");

    // Open stream truncated mid-moov asks for the full moov box before mdat.
    size_t codec_needed = 0;
    require(asr::media::sniff_pipe_audio_codec(
                MediaFamily::kMp4, reinterpret_cast<const uint8_t*>(mp4_aac.data()), 32,
                /*stream_complete=*/false, &codec_needed)
                .empty() &&
                codec_needed == mp4_aac.size(),
            "incomplete open fast-start MP4 requests full moov size");

    // Matroska / WebM EBML header + Segment (unknown size) + Tracks (A_OPUS, TrackType=2).
    auto ebml_vint1 = [](uint8_t payload_len) {
        return static_cast<char>(0x80U | payload_len);
    };
    std::string track_entry;
    track_entry.push_back(static_cast<char>(0x83));  // TrackType
    track_entry.push_back(ebml_vint1(1));
    track_entry.push_back(0x02);                     // 2 = audio
    track_entry.push_back(static_cast<char>(0x86));  // CodecID
    track_entry.push_back(ebml_vint1(6));
    track_entry += "A_OPUS";
    std::string tracks_body;
    tracks_body.push_back(static_cast<char>(0xAE));  // TrackEntry
    tracks_body.push_back(ebml_vint1(static_cast<uint8_t>(track_entry.size())));
    tracks_body += track_entry;
    std::string mkv_hdr("\x1A\x45\xDF\xA3\x84\x42\x86\x81\x01", 9);  // EBML header
    mkv_hdr.append("\x18\x53\x80\x67\xFF", 5);                        // Segment (unknown size)
    mkv_hdr.append("\x16\x54\xAE\x6B", 4);                            // Tracks
    mkv_hdr.push_back(ebml_vint1(static_cast<uint8_t>(tracks_body.size())));
    mkv_hdr += tracks_body;
    require(asr::media::sniff_pipe_audio_codec(
                MediaFamily::kMkv, reinterpret_cast<const uint8_t*>(mkv_hdr.data()),
                mkv_hdr.size()) == "opus",
            "Matroska/WebM EBML Tracks A_OPUS sniffs as opus");
}

}  // namespace

int main() {
    try {
        verify_families();
        verify_rejects();
        verify_filename_fallback();
        verify_stream_layouts();
        verify_family_kinds_and_proto();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("sniff-test passed");
    return 0;
}
