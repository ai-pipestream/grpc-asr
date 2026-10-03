#pragma once

// Shared helpers for the plain-main test suite: an assertion that throws,
// and in-memory media fixtures authored by the tests (no committed
// binaries).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

inline void require(bool condition, const std::string& what) {
    if (!condition) {
        throw std::runtime_error("FAIL: " + what);
    }
}

// Builds a 16-bit PCM WAV in memory: `seconds` of a sine at `frequency`
// Hz (0 = digital silence) at the given sample rate, mono.
inline std::string make_wav(double seconds, double frequency, uint32_t sample_rate = 16000) {
    const uint32_t frames = static_cast<uint32_t>(seconds * sample_rate);
    const uint32_t data_bytes = frames * 2;
    std::string wav;
    wav.reserve(44 + data_bytes);
    auto push_u32 = [&](uint32_t v) {
        wav.push_back(static_cast<char>(v & 0xFF));
        wav.push_back(static_cast<char>((v >> 8) & 0xFF));
        wav.push_back(static_cast<char>((v >> 16) & 0xFF));
        wav.push_back(static_cast<char>((v >> 24) & 0xFF));
    };
    auto push_u16 = [&](uint16_t v) {
        wav.push_back(static_cast<char>(v & 0xFF));
        wav.push_back(static_cast<char>((v >> 8) & 0xFF));
    };
    wav += "RIFF";
    push_u32(36 + data_bytes);
    wav += "WAVEfmt ";
    push_u32(16);
    push_u16(1);  // PCM
    push_u16(1);  // mono
    push_u32(sample_rate);
    push_u32(sample_rate * 2);
    push_u16(2);
    push_u16(16);
    wav += "data";
    push_u32(data_bytes);
    for (uint32_t i = 0; i < frames; i++) {
        double sample = frequency == 0.0
                            ? 0.0
                            : 0.25 * std::sin(2.0 * 3.14159265358979 * frequency * i / sample_rate);
        push_u16(static_cast<uint16_t>(static_cast<int16_t>(sample * 32767.0)));
    }
    return wav;
}

// Builds a 16 kHz mono 16-bit PCM WAV of low white noise from a fixed-seed
// generator, so every run gets the same bytes. whisper's no-speech gate
// decodes a window of it into no segment at all, which digital silence
// does not do (it decodes into a "[BLANK_AUDIO]" segment).
inline std::string make_hiss_wav(double seconds, double amplitude = 0.1) {
    std::string wav = make_wav(seconds, 0.0);
    uint32_t state = 12345;
    for (size_t offset = 44; offset + 1 < wav.size(); offset += 2) {
        state = state * 1664525u + 1013904223u;
        const double unit = static_cast<double>(state >> 8) / 8388608.0 - 1.0;  // [-1, 1)
        const auto sample =
            static_cast<uint16_t>(static_cast<int16_t>(amplitude * unit * 32767.0));
        wav[offset] = static_cast<char>(sample & 0xFF);
        wav[offset + 1] = static_cast<char>(sample >> 8);
    }
    return wav;
}

// Bytes that sniff as mp3 (ID3v2 header) but hold no decodable frame —
// the deterministic "truncated mp3".
inline std::string make_truncated_mp3() {
    std::string mp3("ID3\x04\x00\x00\x00\x00\x00\x0A", 10);
    mp3.append(512, '\x55');
    return mp3;
}

// Reads a whole file; empty when missing.
inline std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

// The ffmpeg that authors media fixtures. The code under test always runs
// the ffmpeg/ffprobe on PATH; GRPC_ASR_TEST_FIXTURE_FFMPEG points fixture
// generation at a separate build with the encoders the shipped one lacks
// (the image's ffmpeg decodes only).
inline std::string fixture_ffmpeg() {
    const char* configured = env_or_null("GRPC_ASR_TEST_FIXTURE_FFMPEG");
    return configured != nullptr ? configured : "ffmpeg";
}

// Standard skip: exit 77 so CTest reports SKIP, not PASS.
inline int skip(const std::string& why) {
    std::println(stderr, "SKIP: {}", why);
    return 77;
}
