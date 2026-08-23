// ByteStream producer/consumer semantics: blocking reads that wake on
// append, short prefixes on early completion, and abort unblocking every
// pending consumer. This is the seam between the upload thread and the
// decode thread, so the blocking contract is the test subject.

#include "media/byte_stream.h"

#include <print>
#include <string>
#include <thread>

#include "fixture.h"

using asr::media::ByteStream;

namespace {

void verify_read_at_present_bytes() {
    ByteStream stream;
    stream.append("abcdef", 6);
    char out[8] = {};
    require(stream.read_at(0, out, 4) == 4, "read_at returns the asked-for bytes");
    require(std::string(out, 4) == "abcd", "read_at copies from the offset");
    require(stream.read_at(4, out, 8) == 2, "read_at returns short at the current end");
    require(std::string(out, 2) == "ef", "the tail bytes are the appended ones");
    require(stream.size() == 6, "size reports appended bytes");
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
    // Offset 2 has no bytes until the producer runs; the read must block
    // and then return them rather than returning 0 for "not yet".
    size_t got = stream.read_at(2, out, 4);
    producer.join();
    require(got == 2, "read past the current end waits for the append");
    require(std::string(out, 2) == "zw", "the awaited bytes are the appended ones");
    require(stream.read_at(4, out, 4) == 0, "read past the end of a complete stream is 0");
}

void verify_prefix_and_completion() {
    ByteStream stream;
    std::thread producer([&] {
        stream.append("RIFF", 4);
        stream.complete();
    });
    // Asks for more than will ever arrive: must return the short size once
    // the stream completes instead of blocking forever.
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
        verify_prefix_and_completion();
        verify_abort_unblocks_consumers();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("byte-stream-test passed");
    return 0;
}
