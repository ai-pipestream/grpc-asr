#include "media/video_demux.h"
#include "media/audio_decoder.h"
#include "cancellation.h"
#include "fixture.h"
#include "native_media_fixture.h"
#include <array>
#include <stop_token>

using asr::media::VideoDemux;
VideoDemux open_native(const std::string& bytes) {
    // Nonexistent tool paths prove the native implementation never launches
    // ffmpeg/ffprobe. Only test-fixture generation below uses a subprocess.
    return VideoDemux(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
                      "/nonexistent/ffmpeg", "/nonexistent/ffprobe", std::chrono::seconds(10));
}
int main() {
    try {
        const auto wav = make_wav(1.0, 440, 48000);
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto media = open_native(wav);
            const auto info = media.probe();
            require(info.has_audio && !info.has_video && info.sample_rate_hz == 48000, "native probe source facts");
            media.open_audio();
            std::array<float, 513> samples;
            size_t count = 0; double energy = 0;
            while (auto n = media.read_audio(samples.data(), samples.size())) {
                count += n;
                for (size_t i = 0; i < n; ++i) {
                    require(std::isfinite(samples[i]) && std::fabs(samples[i]) <= 1, "finite bounded PCM");
                    energy += samples[i] * samples[i];
                }
            }
            media.close_audio();
            require(count == 16000 && energy > 100, "native resampling count and non-silence");
        }
        std::stop_source stopped; stopped.request_stop();
        bool cancelled = false;
        try { auto media = open_native(wav); media.probe(stopped.get_token()); }
        catch (const asr::Cancelled&) { cancelled = true; }
        require(cancelled, "cancel before native probe");
        auto media = open_native(wav);
        std::stop_source during;
        media.open_audio(during.get_token());
        during.request_stop();
        cancelled = false;
        float sample;
        try { media.read_audio(&sample, 1); } catch (const asr::Cancelled&) { cancelled = true; }
        require(cancelled, "cancel during native audio");
        media.cancel_audio(); media.open_audio();
        require(media.read_audio(&sample, 1) == 1, "recovery after cancellation");
        media.close_audio();
        bool rejected = false;
        try { auto bad = open_native("not media"); bad.probe(); }
        catch (const asr::media::DecodeError&) { rejected = true; }
        require(rejected, "invalid native media rejected");
        rejected = false;
        try { auto missing = open_native(""); }
        catch (const asr::media::DecodeError&) { rejected = true; }
        require(rejected, "empty native media rejected");
        rejected = false;
        try { VideoDemux invalid(reinterpret_cast<const uint8_t*>(wav.data()), wav.size(), "", "", std::chrono::milliseconds(0)); }
        catch (const asr::media::ToolError&) { rejected = true; }
        require(rejected, "nonpositive native operation timeout rejected");

        // Generate a tiny video+audio container entirely through a pipe.
        const std::string video = pipe_fixture(
            " -f lavfi -i testsrc2=size=64x48:rate=4"
            " -f lavfi -i sine=frequency=440:sample_rate=16000 -t 2"
            " -c:v mpeg4 -threads:v 1 -c:a aac -f matroska pipe:1");
        auto movie = open_native(video);
        require(movie.probe().has_video && movie.probe().has_audio, "native video and audio tracks");
        size_t frames = 0; uint64_t last = 0;
        movie.extract_keyframes(1, [&](uint64_t ms, uint32_t w, uint32_t h, std::string png) {
            uint32_t pw = 0, ph = 0; asr::media::png_dimensions(png, &pw, &ph);
            require(w == 64 && h == 48 && pw == w && ph == h, "native PNG dimensions");
            require(frames == 0 || ms > last, "native keyframe timestamps advance");
            last = ms; ++frames; return true;
        });
        require(frames == 2, "native keyframe interval count");
        frames = 0;
        movie.extract_keyframes(1, [&](uint64_t, uint32_t, uint32_t, std::string) { ++frames; return false; });
        require(frames == 1, "native keyframe sink cancellation");
        const auto mp4 = trailing_moov_fixture(video);
        require(mp4.find("moov") > mp4.find("mdat"), "fixture has trailing moov");
        auto seekable = open_native(mp4);
        require(seekable.probe().has_audio && seekable.probe().has_video, "trailing-moov MP4 probe");
        seekable.open_audio(); size_t mp4_samples = 0; std::array<float, 1024> block;
        while (auto n = seekable.read_audio(block.data(), block.size())) mp4_samples += n;
        require(mp4_samples >= 32000 && mp4_samples < 35000, "trailing-moov MP4 decodes complete audio");
        for (const auto length : {size_t{8}, mp4.find("moov") - 4}) {
            rejected = false;
            try { auto truncated = open_native(mp4.substr(0, length)); truncated.probe(); }
            catch (const asr::media::DecodeError&) { rejected = true; }
            require(rejected, "truncated MP4 rejected");
        }
        const auto dual = pipe_fixture("-f lavfi -i sine=frequency=220:sample_rate=16000"
            " -f lavfi -i sine=frequency=880:sample_rate=24000 -t 2 -map 0:a -map 1:a"
            " -disposition:a:0 0 -disposition:a:1 default -c:a pcm_s16le -f matroska pipe:1");
        auto tracks = open_native(dual);
        require(tracks.probe().sample_rate_hz == 16000, "probe selects first audio despite second default");
        tracks.open_audio(); size_t crossings = 0; float previous = 0;
        while (auto n = tracks.read_audio(block.data(), block.size())) for (size_t i = 0; i < n; ++i) {
            if (previous <= 0 && block[i] > 0) ++crossings;
            previous = block[i];
        }
        require(crossings > 400 && crossings < 480, "decoded source is first 220Hz track, not default 880Hz track");
        std::println("native-video-demux-test passed");
        return 0;
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what()); return 1;
    }
}
