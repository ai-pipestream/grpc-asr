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

// Standard skip: exit 77 so CTest reports SKIP, not PASS.
inline int skip(const std::string& why) {
    std::println(stderr, "SKIP: {}", why);
    return 77;
}
