// ByteStream producer/consumer semantics: blocking reads that wake on append,
// bounded-wait read_at_for for stall detection, incremental FNV-1a hashing,
// single-copy memfd migration and sealing, short prefixes on early completion,
// and abort unblocking every pending consumer.

#include "media/byte_stream.h"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <print>
#include <string>
#include <thread>

#include "fixture.h"
#include "metrics.h"

using asr::media::ByteStream;

namespace {

uint64_t fnv1a_ref(const std::string& bytes) {
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char b : bytes) {
        hash ^= b;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void verify_read_at_present_bytes() {
    ByteStream stream;
    stream.append("abcdef", 6);
    char out[8] = {};
    require(stream.read_at(0, out, 4) == 4, "read_at returns the asked-for bytes");
    require(std::string(out, 4) == "abcd", "read_at copies from the offset");
    require(stream.read_at(4, out, 8) == 2, "read_at returns short at the current end");
    require(std::string(out, 2) == "ef", "the tail bytes are the appended ones");
    require(stream.size() == 6, "size reports appended bytes");
    require(stream.available_from(2) == 4, "available_from reports buffered tail");
    require(stream.available_from(6) == 0, "available_from at end is 0");
    require(stream.content_hash() == fnv1a_ref("abcdef"), "incremental FNV-1a matches reference");
    require(!stream.is_complete() && !stream.is_aborted(), "stream still open");
}

void verify_read_blocks_until_append() {
    ByteStream stream;
    stream.append("xy", 2);
    char out[4] = {};
    std::thread producer([&] {
        stream.append("zw", 2);
        stream.complete();
    });
    size_t got = stream.read_at(2, out, 4);
    producer.join();
    require(got == 2, "read past the current end waits for the append");
    require(std::string(out, 2) == "zw", "the awaited bytes are the appended ones");
    require(stream.read_at(4, out, 4) == 0, "read past the end of a complete stream is 0");
    require(stream.content_hash() == fnv1a_ref("xyzw"), "multi-chunk hash matches reference");
}

void verify_read_at_for_timeout_and_wakeup() {
    ByteStream stream;
    stream.append("ab", 2);
    char out[4] = {};
    const auto r0 = stream.read_at_for(0, out, 4, std::chrono::milliseconds(50));
    require(!r0.timed_out && r0.bytes == 2 && std::string(out, 2) == "ab",
            "present bytes return without timeout");

    const auto r1 = stream.read_at_for(2, out, 4, std::chrono::milliseconds(30));
    require(r1.timed_out && r1.bytes == 0, "missing bytes time out while open");

    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stream.append("cd", 2);
        stream.complete();
    });
    const auto r2 = stream.read_at_for(2, out, 4, std::chrono::milliseconds(500));
    producer.join();
    require(!r2.timed_out && r2.bytes == 2 && std::string(out, 2) == "cd",
            "arrival within timeout wakes read_at_for");

    const auto r3 = stream.read_at_for(4, out, 4, std::chrono::milliseconds(50));
    require(!r3.timed_out && r3.bytes == 0, "completed EOF returns {0, false}");
}

void verify_memfd_mode_and_gauges() {
    const auto base = asr::metrics_summary();
    {
        ByteStream stream;
        stream.append("hello ", 6);
        const auto s1 = asr::metrics_summary();
        require(s1.upload_bytes_resident == base.upload_bytes_resident + 6,
                "string upload bytes tracked");
        require(s1.memfd_bytes_resident == base.memfd_bytes_resident, "memfd bytes still unchanged");

        stream.switch_to_memfd();
        require(stream.is_memfd_backed() && stream.memfd() >= 0, "memfd allocated");
        require(stream.completed_bytes().empty(), "heap buffer freed on switch_to_memfd");

        const auto s2 = asr::metrics_summary();
        require(s2.upload_bytes_resident == base.upload_bytes_resident,
                "upload bytes migrated out of heap gauge");
        require(s2.memfd_bytes_resident == base.memfd_bytes_resident + 6,
                "migrated prefix counted in memfd gauge");

        stream.append("world!", 6);
        const auto s3 = asr::metrics_summary();
        require(s3.memfd_bytes_resident == base.memfd_bytes_resident + 12,
                "subsequent appends land directly in memfd");
        require(stream.prefix_bytes(5) == "hello", "prefix_bytes reads from memfd");

        char out[16] = {};
        require(stream.read_at(0, out, 16) == 12 && std::string(out, 12) == "hello world!",
                "read_at reads full migrated + appended memfd content");
        require(stream.content_hash() == fnv1a_ref("hello world!"),
                "hash preserved across memfd switch");

        stream.complete();
        const int seals = ::fcntl(stream.memfd(), F_GET_SEALS);
        const int wanted = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL;
        require(seals >= 0 && (seals & wanted) == wanted, "complete() seals the memfd");
        require(::pwrite(stream.memfd(), "x", 1, 0) < 0, "sealed memfd rejects writes");
    }
    const auto after = asr::metrics_summary();
    require(after.upload_bytes_resident == base.upload_bytes_resident &&
                after.memfd_bytes_resident == base.memfd_bytes_resident,
            "destructor releases resident byte gauges");
}

void verify_prefix_and_completion() {
    ByteStream stream;
    std::thread producer([&] {
        stream.append("RIFF", 4);
        stream.complete();
    });
    size_t prefix = stream.wait_for_prefix(16);
    producer.join();
    require(prefix == 4, "wait_for_prefix returns short on early completion");
    stream.wait_complete();
    require(stream.is_complete(), "complete flag visible after wait_complete");
    require(stream.completed_bytes() == "RIFF", "completed_bytes is the whole upload");
}

void verify_abort_unblocks_consumers() {
    ByteStream stream;
    stream.append("abc", 3);
    char out[4] = {};
    std::thread consumer([&] {
        require(stream.read_at(3, out, 4) == 0, "pending read returns 0 on abort");
    });
    std::thread waiter([&] {
        stream.wait_complete();
        require(stream.is_aborted(), "wait_complete returns on abort");
    });
    stream.abort();
    consumer.join();
    waiter.join();
    require(stream.read_at(0, out, 4) == 0, "reads after abort return 0 even for present bytes");
    require(stream.wait_for_prefix(16) == 3, "wait_for_prefix stops waiting after abort");
}

}  // namespace

int main() {
    try {
        verify_read_at_present_bytes();
        verify_read_blocks_until_append();
        verify_read_at_for_timeout_and_wakeup();
        verify_memfd_mode_and_gauges();
        verify_prefix_and_completion();
        verify_abort_unblocks_consumers();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("byte-stream-test passed");
    return 0;
}
