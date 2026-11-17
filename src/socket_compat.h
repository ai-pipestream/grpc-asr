#pragma once

#include <cerrno>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace asr {
// Linux sets CLOEXEC atomically. Darwin has no SOCK_CLOEXEC/accept4;
// native media never spawns decoder children, but still closes on exec.
inline int cloexec_socket(int domain, int type, int protocol) {
#ifdef __linux__
    return ::socket(domain, type | SOCK_CLOEXEC, protocol);
#else
    const int fd = ::socket(domain, type, protocol);
    if (fd >= 0 && ::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
        const int error = errno; ::close(fd); errno = error; return -1;
    }
    return fd;
#endif
}
inline int cloexec_accept(int server) {
#ifdef __linux__
    return ::accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
#else
    const int fd = ::accept(server, nullptr, nullptr);
    if (fd >= 0 && ::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
        const int error = errno; ::close(fd); errno = error; return -1;
    }
    return fd;
#endif
}
inline ssize_t send_no_signal(int fd, const void* data, size_t size) {
#ifdef __APPLE__
    const int enabled = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) < 0) return -1;
    return ::send(fd, data, size, 0);
#else
    return ::send(fd, data, size, MSG_NOSIGNAL);
#endif
}
} // namespace asr
