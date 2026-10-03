// Container sniffing over hand-authored magic bytes.

#include "media/sniff.h"

#include <print>

#include "fixture.h"

using asr::media::MediaFamily;
using asr::media::sniff;

namespace {

MediaFamily sniff_str(const std::string& bytes) {
    return sniff(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
}

void verify_families() {
    require(sniff_str(make_wav(0.01, 0)) == MediaFamily::kWav, "wav sniffs as wav");
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
}

void verify_rejects() {
    require(sniff_str("plain text, not media at all") == MediaFamily::kUnknown,
            "text is unknown");
    require(sniff_str("") == MediaFamily::kUnknown, "empty is unknown");
    require(sniff_str("RI") == MediaFamily::kUnknown, "short prefix is unknown");
    require(sniff_str("RIFFxxxxAVI ") == MediaFamily::kUnknown,
            "RIFF without WAVE is unknown (avi is not audio)");
    // ADTS AAC shares the 12-bit sync run but carries layer 00, which MPEG
    // audio reserves; it used to sniff as mp3 and fail inside the mp3
    // decoder instead of being refused as an unsupported container.
    require(sniff_str(std::string("\xFF\xF1\x50\x80", 4)) == MediaFamily::kUnknown,
            "MPEG-4 ADTS AAC is not mp3");
    require(sniff_str(std::string("\xFF\xF9\x50\x80", 4)) == MediaFamily::kUnknown,
            "MPEG-2 ADTS AAC is not mp3");
    require(sniff_str(std::string("\xFF\xEB\x90\x00", 4)) == MediaFamily::kUnknown,
            "a frame with the reserved MPEG version is not mp3");
}

void verify_family_kinds() {
    using asr::media::decodes_in_process;
    using asr::media::demuxes_with_ffmpeg;
    for (MediaFamily family : {MediaFamily::kWav, MediaFamily::kMp3, MediaFamily::kFlac}) {
        require(decodes_in_process(family) && !demuxes_with_ffmpeg(family),
                std::string(asr::media::family_name(family)) + " decodes in process");
    }
    // miniaudio decodes neither Vorbis nor Opus here: an ogg that took the
    // in-process path was sniffed, then always refused as undecodable.
    for (MediaFamily family : {MediaFamily::kOgg, MediaFamily::kMp4, MediaFamily::kMkv}) {
        require(demuxes_with_ffmpeg(family) && !decodes_in_process(family),
                std::string(asr::media::family_name(family)) + " goes through ffmpeg");
    }
    require(!decodes_in_process(MediaFamily::kUnknown) &&
                !demuxes_with_ffmpeg(MediaFamily::kUnknown),
            "unknown takes neither path");
}

void verify_mimetypes() {
    // The Document origin is stamped from these, so every family a sniff
    // can return names a real type and nothing falls through to the
    // catch-all by accident.
    require(asr::media::family_mimetype(MediaFamily::kWav) == "audio/wav", "wav mimetype");
    require(asr::media::family_mimetype(MediaFamily::kMp3) == "audio/mpeg", "mp3 mimetype");
    require(asr::media::family_mimetype(MediaFamily::kFlac) == "audio/flac", "flac mimetype");
    require(asr::media::family_mimetype(MediaFamily::kOgg) == "audio/ogg", "ogg mimetype");
    require(asr::media::family_mimetype(MediaFamily::kMp4) == "video/mp4", "mp4 mimetype");
    require(asr::media::family_mimetype(MediaFamily::kMkv) == "video/x-matroska",
            "mkv mimetype");
    require(asr::media::family_mimetype(MediaFamily::kUnknown) == "application/octet-stream",
            "an unsniffed container claims nothing");
}

}  // namespace

int main() {
    try {
        verify_families();
        verify_rejects();
        verify_family_kinds();
        verify_mimetypes();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("sniff-test passed");
    return 0;
}
