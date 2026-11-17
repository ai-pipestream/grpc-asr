#pragma once

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace asr::media {
// On macOS callers serialize pipe creation with fork, or call before any
// worker threads start. Linux atomically applies CLOEXEC with pipe2.
inline int pipe_cloexec(int fds[2]) {
#ifdef __APPLE__
    if (::pipe(fds) != 0) return -1;
    if (::fcntl(fds[0], F_SETFD, FD_CLOEXEC) == 0 &&
        ::fcntl(fds[1], F_SETFD, FD_CLOEXEC) == 0) return 0;
    const int saved = errno;
    ::close(fds[0]);
    ::close(fds[1]);
    errno = saved;
    return -1;
#else
    return ::pipe2(fds, O_CLOEXEC);
#endif
}
}  // namespace asr::media
