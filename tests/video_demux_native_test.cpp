#include "media/video_demux.h"
#include "media/audio_decoder.h"
#include "cancellation.h"
#include "fixture.h"
#include "native_media_fixture.h"
#include "media/native_memory_seek.h"
#include <array>
#include <stop_token>

using asr::media::VideoDemux;
VideoDemux open_native(const std::string& bytes) {
    // Nonexistent tool paths prove the native implementation never launches
    // ffmpeg/ffprobe. Only test-fixture generation below uses a subprocess.
    return VideoDemux(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
                      "/nonexistent/ffmpeg", "/nonexistent/ffprobe", std::chrono::seconds(10));
}
// Check the encoded fixture, not just the ffmpeg command: muxers may rebase
// timestamps. These assertions prevent a nominal negative-origin test from
// silently becoming another zero-origin test.
void verify_fixture_origin(const std::string& bytes, int origin, int delayed_track) {
    struct FixtureInput {
        const std::string& bytes;
        size_t offset = 0;
        AVIOContext* io = nullptr;
        AVFormatContext* format = nullptr;
        ~FixtureInput() {
            avformat_close_input(&format);
            if (io) av_freep(&io->buffer);
            avio_context_free(&io);
        }
    } input{bytes};
    auto* buffer = static_cast<uint8_t*>(av_malloc(4096));
    require(buffer != nullptr, "timeline fixture buffer");
    input.io = avio_alloc_context(buffer, 4096, 0, &input,
        [](void* opaque, uint8_t* dst, int size) {
            auto& in = *static_cast<FixtureInput*>(opaque);
            const size_t count = std::min(static_cast<size_t>(size), in.bytes.size() - in.offset);
            if (!count) return AVERROR_EOF;
            std::memcpy(dst, in.bytes.data() + in.offset, count); in.offset += count;
            return static_cast<int>(count);
        }, nullptr, nullptr);
    if (!input.io) av_free(buffer);
    require(input.io != nullptr, "timeline fixture AVIO");
    input.format = avformat_alloc_context();
    require(input.format != nullptr, "timeline fixture format");
    input.format->pb = input.io; input.format->flags |= AVFMT_FLAG_CUSTOM_IO;
    require(avformat_open_input(&input.format, nullptr, nullptr, nullptr) == 0 &&
            avformat_find_stream_info(input.format, nullptr) >= 0, "timeline fixture probe");
    require(input.format->start_time == static_cast<int64_t>(origin) * AV_TIME_BASE,
            "muxed fixture retains authored container origin");
    require(input.format->nb_streams == 2, "timeline fixture has exactly two tracks");
    for (unsigned i = 0; i < 2; ++i) {
        const auto* stream = input.format->streams[i];
        const bool delayed = (delayed_track == 1 && stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) ||
                             (delayed_track == 2 && stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO);
        require(stream->start_time != AV_NOPTS_VALUE &&
                av_rescale_q(stream->start_time, stream->time_base, AVRational{1, 1000}) ==
                    static_cast<int64_t>(origin) * 1000 + (delayed ? 500 : 0),
                "muxed fixture retains authored track offset");
    }
}
// PCM avoids encoder priming: the authored stream offsets are exact. The
// same assertions must hold after translating the entire container timeline.
void verify_timeline_origins() {
    int failures = 0;
    for (const int origin : {10, -1, 0}) {
        for (const int delayed_track : {0, 1, 2}) { // none, video, audio
            const std::string label = "origin=" + std::to_string(origin) +
                " delayed_track=" + std::to_string(delayed_track);
            try {
                const auto bytes = pipe_fixture(
                    " -f lavfi -i testsrc2=size=64x48:rate=4:duration=2"
                    " -f lavfi -i sine=frequency=440:sample_rate=16000:duration=2"
                    " -map 0:v -map 1:a -vf setpts=PTS+" + std::string(delayed_track == 1 ? "0.5" : "0") + "/TB"
                    " -af asetpts=PTS+" + std::string(delayed_track == 2 ? "0.5" : "0") + "/TB"
                    " -c:v mpeg4 -threads:v 1 -fps_mode passthrough -c:a pcm_s16le"
                    " -output_ts_offset " + std::to_string(origin) +
                    " -avoid_negative_ts disabled -f matroska pipe:1");
                verify_fixture_origin(bytes, origin, delayed_track);
                auto media = open_native(bytes);
                std::vector<uint64_t> times;
                media.extract_keyframes(1, [&](uint64_t ms, uint32_t, uint32_t, std::string) {
                    times.push_back(ms); return true;
                });
                const uint64_t video_start = delayed_track == 1 ? 500 : 0;
                require(times == std::vector<uint64_t>{video_start, video_start + 1000},
                        label + " keyframes preserve media-relative video offset");
                if (delayed_track == 2) {
                    std::stop_source stop;
                    media.open_audio(stop.get_token());
                    float zero = 1;
                    require(media.read_audio(&zero, 1) == 1 && zero == 0, label + " streamed leading silence");
                    stop.request_stop();
                    bool cancelled = false;
                    try { media.read_audio(&zero, 1); }
                    catch (const asr::Cancelled&) { cancelled = true; }
                    require(cancelled, label + " cancellation during leading silence");
                    media.cancel_audio();
                }
                media.open_audio(); // also resets a partially consumed silence gap

                std::array<float, 513> samples;
                const size_t silence = delayed_track == 2 ? 8000 : 0;
                size_t count = 0; double energy = 0;
                while (auto n = media.read_audio(samples.data(), samples.size())) {
                    for (size_t i = 0; i < n; ++i) {
                        if (count + i < silence) require(samples[i] == 0, label + " leading audio silence");
                        else energy += samples[i] * samples[i];
                    }
                    count += n;
                    require(count <= silence + 32000, label + " bounded PCM duration");
                }
                require(count == silence + 32000 && energy > 100,
                        label + " audio preserves media-relative start and duration");
                media.close_audio();
                std::println("timeline PASS {}", label);
            } catch (const std::exception& error) {
                ++failures; std::println(stderr, "timeline {}", error.what());
            }
        }
    }
    require(failures == 0, "timeline origin regression cases: " + std::to_string(failures) + " failed");
}
int main(int argc, char**) {
    try {
        verify_timeline_origins();
        if (argc > 1) return 0;
        size_t position = 7;
        require(asr::media::native_memory_seek(100, position, 0, AVSEEK_SIZE | AVSEEK_FORCE) == 100 && position == 7,
                "forced size query returns length without moving cursor");
        require(asr::media::native_memory_seek(100, position, -2, SEEK_END | AVSEEK_FORCE) == 98,
                "forced end-relative seek");
        require(asr::media::native_memory_seek(100, position, 3, SEEK_CUR) < 0 && position == 98,
                "out-of-bounds seek preserves cursor");
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
