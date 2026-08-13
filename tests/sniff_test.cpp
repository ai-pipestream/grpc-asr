// Container sniffing over hand-authored magic bytes.

#include "media/sniff.h"

#include <iostream>

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

}  // namespace

int main() {
    try {
        verify_families();
        verify_rejects();
        verify_family_kinds();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "sniff-test passed\n";
    return 0;
}
