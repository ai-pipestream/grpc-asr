#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>

#include "metrics.h"

namespace asr::media {

// A growing in-memory byte buffer bridging the upload thread and the decode
// thread, so transcription starts while the media is still uploading.
//
// Starts backed by an in-memory std::string (counted under
// grpc_asr_upload_bytes_resident). Callers that need a seekable anonymous file
// descriptor (trailing-moov MP4, keyframe extraction, or multi-channel ffmpeg
// demux) call switch_to_memfd(), which migrates any already-buffered prefix
// into a sealable memfd, releases the heap buffer, writes subsequent append()
// chunks directly into the memfd, and seals the memfd on complete() so at most
// one copy of the media ever lives in RAM.
class ByteStream {
  public:
    struct ReadResult {
        size_t bytes = 0;
        bool timed_out = false;
    };

    ByteStream() = default;

    ~ByteStream() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (memfd_ >= 0) {
            metrics_add_memfd_bytes(-static_cast<int64_t>(total_size_));
            ::close(memfd_);
            memfd_ = -1;
        } else if (total_size_ > 0) {
            metrics_add_upload_bytes(-static_cast<int64_t>(total_size_));
        }
    }

    ByteStream(const ByteStream&) = delete;
    ByteStream& operator=(const ByteStream&) = delete;

    // Producer side: appends the next chunk of encoded media and updates the
    // incremental FNV-1a 64 content hash.
    void append(const char* data, size_t size) {
        if (size == 0) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < size; ++i) {
                hash_ ^= static_cast<unsigned char>(data[i]);
                hash_ *= 1099511628211ULL;
            }
            if (memfd_ >= 0) {
                write_memfd_locked(total_size_, data, size);
                total_size_ += size;
                metrics_add_memfd_bytes(static_cast<int64_t>(size));
            } else {
                bytes_.append(data, size);
                total_size_ += size;
                metrics_add_upload_bytes(static_cast<int64_t>(size));
            }
        }
        grew_.notify_all();
    }

    // Migrates buffered bytes to an anonymous sealable memfd and routes future
    // append() calls directly into it. Idempotent.
    void switch_to_memfd() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (memfd_ >= 0) {
            return;
        }
        const int fd =
            static_cast<int>(::memfd_create("grpc-asr-media", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        if (fd < 0) {
            throw std::runtime_error("memfd_create failed");
        }
        try {
            if (!bytes_.empty()) {
                size_t written = 0;
                while (written < bytes_.size()) {
                    const ssize_t n = ::pwrite(fd, bytes_.data() + written, bytes_.size() - written,
                                               static_cast<off_t>(written));
                    if (n <= 0) {
                        throw std::runtime_error("writing media to memfd failed");
                    }
                    written += static_cast<size_t>(n);
                }
            }
            if (complete_) {
                if (::fcntl(fd, F_ADD_SEALS,
                            F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
                    throw std::runtime_error(std::string("sealing the media memfd failed: ") +
                                             std::strerror(errno));
                }
                sealed_ = true;
            }
        } catch (...) {
            ::close(fd);
            throw;
        }
        memfd_ = fd;
        std::string().swap(bytes_);
        if (total_size_ > 0) {
            metrics_add_upload_bytes(-static_cast<int64_t>(total_size_));
            metrics_add_memfd_bytes(static_cast<int64_t>(total_size_));
        }
    }

    // Producer side: no more bytes will arrive; reads past the end now return
    // short instead of blocking. Seals the memfd when memfd-backed.
    void complete() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            complete_ = true;
            if (memfd_ >= 0 && !sealed_) {
                if (::fcntl(memfd_, F_ADD_SEALS,
                            F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
                    throw std::runtime_error(std::string("sealing the media memfd failed: ") +
                                             std::strerror(errno));
                }
                sealed_ = true;
            }
        }
        grew_.notify_all();
    }

    // Producer side: the upload died. Pending and future reads return 0 so the
    // decoder fails out instead of waiting forever.
    void abort() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            aborted_ = true;
        }
        grew_.notify_all();
    }

    // Consumer side: copies up to size bytes from offset, blocking until at
    // least one byte past offset exists, the stream finishes, or `stop` fires.
    // Returns 0 at end of data, on stop, or after abort().
    size_t read_at(uint64_t offset, void* out, size_t size, std::stop_token stop = {}) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto ready = [&] { return aborted_ || complete_ || total_size_ > offset; };
        if (stop.stop_possible()) {
            if (!grew_.wait(lock, stop, ready)) {
                return 0;
            }
        } else {
            grew_.wait(lock, ready);
        }
        if (aborted_ || offset >= total_size_) {
            return 0;
        }
        return copy_out_locked(offset, out, size);
    }

    // Bounded-wait consumer read for stall-aware window filling. When no byte
    // past `offset` arrives within `timeout` and the stream is still open,
    // returns {0, /*timed_out=*/true}.
    ReadResult read_at_for(uint64_t offset, void* out, size_t size,
                           std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool ready = grew_.wait_for(
            lock, timeout, [&] { return aborted_ || complete_ || total_size_ > offset; });
        if (!ready) {
            return {0, true};
        }
        if (aborted_ || offset >= total_size_) {
            return {0, false};
        }
        return {copy_out_locked(offset, out, size), false};
    }

    // Non-blocking count of bytes currently readable from `offset`.
    uint64_t available_from(uint64_t offset) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (aborted_ || offset >= total_size_) {
            return 0;
        }
        return total_size_ - offset;
    }

    // Blocks up to `timeout` until at least `min_bytes` exist starting at
    // `offset`, or the stream finishes or aborts. Returns false when the
    // stream is still open and timed out short of `min_bytes`.
    bool wait_for_available(uint64_t offset, size_t min_bytes,
                            std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return grew_.wait_for(lock, timeout, [&] {
            return aborted_ || complete_ || total_size_ >= offset + min_bytes;
        });
    }

    // Consumer side: blocks until at least `size` total bytes arrived (or the
    // stream finished shorter). Used to sniff the container and ISO BMFF box
    // layout from the leading bytes.
    size_t wait_for_prefix(size_t size) {
        std::unique_lock<std::mutex> lock(mutex_);
        grew_.wait(lock, [&] { return aborted_ || complete_ || total_size_ >= size; });
        return total_size_ < size ? static_cast<size_t>(total_size_) : size;
    }

    // Copies up to `max_bytes` from offset 0 regardless of backing mode.
    std::string prefix_bytes(size_t max_bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t count =
            total_size_ < max_bytes ? static_cast<size_t>(total_size_) : max_bytes;
        std::string out(count, '\0');
        if (count > 0) {
            copy_out_locked(0, out.data(), count);
        }
        return out;
    }

    // Consumer side: blocks until the producer called complete() or abort().
    void wait_complete() {
        std::unique_lock<std::mutex> lock(mutex_);
        grew_.wait(lock, [&] { return complete_ || aborted_; });
    }

    bool is_complete() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return complete_;
    }

    bool is_aborted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return aborted_;
    }

    uint64_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_size_;
    }

    // Incremental FNV-1a 64 hash of every byte passed to append().
    uint64_t content_hash() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return hash_;
    }

    // Sealed memfd descriptor when switch_to_memfd() + complete() ran, else -1.
    int memfd() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return memfd_;
    }

    bool is_memfd_backed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return memfd_ >= 0;
    }

    // Consumer-side view of the raw bytes when string-backed and complete().
    const std::string& completed_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bytes_;
    }

  private:
    void write_memfd_locked(uint64_t offset, const char* data, size_t size) {
        size_t written = 0;
        while (written < size) {
            const ssize_t n = ::pwrite(memfd_, data + written, size - written,
                                       static_cast<off_t>(offset + written));
            if (n <= 0) {
                throw std::runtime_error("writing media to memfd failed");
            }
            written += static_cast<size_t>(n);
        }
    }

    size_t copy_out_locked(uint64_t offset, void* out, size_t size) const {
        const uint64_t available = total_size_ - offset;
        const size_t count =
            size < available ? size : static_cast<size_t>(available);
        if (count == 0) {
            return 0;
        }
        if (memfd_ >= 0) {
            size_t got = 0;
            char* dst = static_cast<char*>(out);
            while (got < count) {
                const ssize_t n =
                    ::pread(memfd_, dst + got, count - got, static_cast<off_t>(offset + got));
                if (n <= 0) {
                    break;
                }
                got += static_cast<size_t>(n);
            }
            return got;
        }
        bytes_.copy(static_cast<char*>(out), count, static_cast<size_t>(offset));
        return count;
    }

    mutable std::mutex mutex_;
    std::condition_variable_any grew_;
    std::string bytes_;
    int memfd_ = -1;
    bool sealed_ = false;
    uint64_t total_size_ = 0;
    uint64_t hash_ = 14695981039346656037ULL;
    bool complete_ = false;
    bool aborted_ = false;
};

}  // namespace asr::media
