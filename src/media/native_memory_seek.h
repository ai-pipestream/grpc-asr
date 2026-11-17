#pragma once
#include <cstddef>
#include <cstdint>
#include <cerrno>
extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
}
namespace asr::media {
inline int64_t native_memory_seek(size_t size, size_t& position, int64_t offset, int whence) {
    whence &= ~AVSEEK_FORCE;
    if (whence == AVSEEK_SIZE) return static_cast<int64_t>(size);
    int64_t base = 0;
    if (whence == SEEK_CUR) base = static_cast<int64_t>(position);
    else if (whence == SEEK_END) base = static_cast<int64_t>(size);
    else if (whence != SEEK_SET) return AVERROR(EINVAL);
    if (offset < -base || offset > static_cast<int64_t>(size) - base) return AVERROR(EINVAL);
    position = static_cast<size_t>(base + offset);
    return static_cast<int64_t>(position);
}
}
