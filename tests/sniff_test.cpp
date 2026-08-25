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
}

void verify_family_kinds() {
    require(asr::media::is_audio_family(MediaFamily::kWav), "wav is audio");
    require(asr::media::is_audio_family(MediaFamily::kOgg), "ogg is audio");
    require(asr::media::is_video_family(MediaFamily::kMp4), "mp4 is video");
    require(asr::media::is_video_family(MediaFamily::kMkv), "mkv is video");
    require(!asr::media::is_audio_family(MediaFamily::kMp4), "mp4 is not the audio path");
    require(!asr::media::is_video_family(MediaFamily::kUnknown), "unknown is neither");
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
