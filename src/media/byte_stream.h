#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace asr::media {

// A growing in-memory byte buffer bridging the upload thread and the
// decode thread, so transcription starts while the media is still
// uploading. The producer appends chunks and eventually calls complete()
// (clean half-close) or abort() (client gone); consumers block in
// read_at() until the bytes they need exist or no more ever will.
class ByteStream {
  public:
    // Producer side: appends the next chunk of encoded media.
    void append(const char* data, size_t size) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            bytes_.append(data, size);
        }
        grew_.notify_all();
    }

    // Producer side: no more bytes will arrive; reads past the end now
    // return short instead of blocking.
    void complete() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            complete_ = true;
        }
        grew_.notify_all();
    }

    // Producer side: the upload died. Pending and future reads return 0
    // so the decoder fails out instead of waiting forever.
    void abort() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            aborted_ = true;
        }
        grew_.notify_all();
    }

    // Consumer side: copies up to size bytes from offset, blocking until
    // at least one byte past offset exists or the stream is finished.
    // Returns 0 at end of data or after abort().
    size_t read_at(uint64_t offset, void* out, size_t size) {
        std::unique_lock<std::mutex> lock(mutex_);
        grew_.wait(lock, [&] { return aborted_ || complete_ || bytes_.size() > offset; });
        if (aborted_ || offset >= bytes_.size()) {
            return 0;
        }
        size_t available = bytes_.size() - static_cast<size_t>(offset);
        size_t count = size < available ? size : available;
        bytes_.copy(static_cast<char*>(out), count, static_cast<size_t>(offset));
        return count;
    }

    // Consumer side: blocks until at least `size` total bytes arrived (or
    // the stream finished shorter). Used to sniff the container from the
    // first bytes.
    size_t wait_for_prefix(size_t size) {
        std::unique_lock<std::mutex> lock(mutex_);
        grew_.wait(lock,
                   [&] { return aborted_ || complete_ || bytes_.size() >= size; });
        return bytes_.size() < size ? bytes_.size() : size;
    }

    // Consumer side: blocks until the producer called complete() or
    // abort(). The video path needs the whole file before ffmpeg can
    // seek it.
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
        return bytes_.size();
    }

    // The consumer-side view of the raw bytes. Only safe once complete();
    // the video path uses it to hand the whole file to the memfd.
    const std::string& completed_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bytes_;
    }

  private:
    mutable std::mutex mutex_;
    std::condition_variable grew_;
    std::string bytes_;
    bool complete_ = false;
    bool aborted_ = false;
};

}  // namespace asr::media
