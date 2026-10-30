#include "http.h"
#include "registry.h"
#include <curl/curl.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>

namespace media {
namespace {
void initialize() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) throw std::runtime_error("libcurl initialization failed");
  });
}
bool public_address(const curl_sockaddr& address) {
  if (address.family == AF_INET) {
    const auto ip = ntohl(reinterpret_cast<const sockaddr_in*>(&address.addr)->sin_addr.s_addr);
    const auto first = ip >> 24;
    return first != 0 && first != 10 && first != 127 && first < 224 &&
      (ip & 0xffff0000U) != 0xa9fe0000U && (ip & 0xfff00000U) != 0xac100000U &&
      (ip & 0xffff0000U) != 0xc0a80000U && (ip & 0xffc00000U) != 0x64400000U &&
      (ip & 0xfffe0000U) != 0xc6120000U && (ip & 0xffffff00U) != 0xc0000000U &&
      (ip & 0xffffff00U) != 0xc0000200U && (ip & 0xffffff00U) != 0xc6336400U &&
      (ip & 0xffffff00U) != 0xcb007100U && (ip & 0xffffff00U) != 0xc0586300U;
  }
  if (address.family == AF_INET6) {
    const auto* ip = reinterpret_cast<const sockaddr_in6*>(&address.addr)->sin6_addr.s6_addr;
    return (ip[0] & 0xe0) == 0x20 && !(ip[0] == 0x20 && ip[1] == 0x02) &&
      !(ip[0] == 0x20 && ip[1] == 0x01 && ip[2] == 0 && ip[3] == 0) &&
      !(ip[0] == 0x20 && ip[1] == 0x01 && ip[2] == 0x0d && ip[3] == 0xb8);
  }
  return false;
}
std::string part(CURLU* url, CURLUPart key, unsigned flags = 0) {
  char* raw = nullptr;
  const auto result = curl_url_get(url, key, &raw, flags);
  if (result != CURLUE_OK) return {};
  std::unique_ptr<char, decltype(&curl_free)> value(raw, curl_free);
  return value.get();
}
bool present(CURLU* url, CURLUPart key) {
  char* value = nullptr;
  const bool found = curl_url_get(url, key, &value, 0) == CURLUE_OK;
  curl_free(value);
  return found;
}
}
std::string HttpInput::origin(const std::string& input) {
  initialize();
  if (input.empty() || input.size() > 8192 || input.find('\0') != std::string::npos || input.find('#') != std::string::npos)
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid HTTP URL");
  std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
  if (!url || curl_url_set(url.get(), CURLUPART_URL, input.c_str(), 0) != CURLUE_OK)
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid HTTP URL");
  auto scheme = part(url.get(), CURLUPART_SCHEME), host = part(url.get(), CURLUPART_HOST);
  const auto authority_start = input.find("://");
  const auto authority_end = input.find_first_of("/?", authority_start == std::string::npos ? 0 : authority_start + 3);
  const auto at = input.find('@', authority_start == std::string::npos ? 0 : authority_start + 3);
  if ((scheme != "http" && scheme != "https") || host.empty() ||
      (at != std::string::npos && (authority_end == std::string::npos || at < authority_end)) ||
      present(url.get(), CURLUPART_USER) || present(url.get(), CURLUPART_PASSWORD) ||
      present(url.get(), CURLUPART_FRAGMENT))
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "HTTP URL scheme or credentials are not permitted");
  std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) { return std::tolower(c); });
  return scheme + "://" + host + ":" + part(url.get(), CURLUPART_PORT, CURLU_DEFAULT_PORT);
}
struct HttpInput::Impl {
  CURL* easy = nullptr;
  CURLM* multi = nullptr;
  HttpPolicy policy;
  std::function<bool()> cancelled;
  std::deque<std::string> pending;
  size_t bytes = 0, offset = 0, header_bytes = 0;
  bool paused = false, finished = false, added = false;
  bool address_rejected = false;
  CURLcode result = CURLE_OK;
  std::exception_ptr error;
  static constexpr size_t budget = 64 * 1024;
  ~Impl() {
    if (added) curl_multi_remove_handle(multi, easy);
    if (easy) curl_easy_cleanup(easy);
    if (multi) curl_multi_cleanup(multi);
  }
  static size_t body(char* data, size_t size, size_t count, void* opaque) noexcept {
    auto& self = *static_cast<Impl*>(opaque);
    try {
      const size_t n = size * count;
      if (!n) return 0;
      long status = 0; curl_easy_getinfo(self.easy, CURLINFO_RESPONSE_CODE, &status);
      if (status < 200 || status >= 300) throw Failure(grpc::StatusCode::FAILED_PRECONDITION, "HTTP source did not return a successful response");
      if (n > budget) throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "HTTP callback exceeds buffer limit");
      if (n > budget - self.bytes) { self.paused = true; return CURL_WRITEFUNC_PAUSE; }
      self.pending.emplace_back(data, n); self.bytes += n; return n;
    } catch (...) { self.error = std::current_exception(); return 0; }
  }
  static size_t header(char* data, size_t size, size_t count, void* opaque) noexcept {
    auto& self = *static_cast<Impl*>(opaque);
    try {
      const size_t n = size * count;
      if (n > budget - self.header_bytes) throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "HTTP headers exceed limit");
      self.header_bytes += n;
      std::string line(data, n);
      std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) { return std::tolower(c); });
      if (line.starts_with("content-encoding:")) {
        auto encoding = line.substr(17);
        std::erase_if(encoding, [](unsigned char c) { return std::isspace(c); });
        if (!encoding.empty() && encoding != "identity")
          throw Failure(grpc::StatusCode::UNIMPLEMENTED, "encoded HTTP response entities are not supported");
      }
      return n;
    } catch (...) { self.error = std::current_exception(); return 0; }
  }
  static curl_socket_t socket(void* opaque, curlsocktype purpose, curl_sockaddr* address) noexcept {
    auto& self = *static_cast<Impl*>(opaque);
    if (purpose != CURLSOCKTYPE_IPCXN || (!self.policy.allow_private_addresses && !public_address(*address))) {
      self.address_rejected = true;
      return CURL_SOCKET_BAD;
    }
    return ::socket(address->family, address->socktype, address->protocol);
  }
  void check() {
    if (error) std::rethrow_exception(error);
    if (cancelled()) throw Failure(grpc::StatusCode::CANCELLED, "HTTP source cancelled");
  }
};
HttpInput::HttpInput(std::string url, HttpPolicy policy, std::function<bool()> cancelled)
    : impl_(std::make_unique<Impl>()) {
  const auto source_origin = origin(url);
  if (!policy.allowed_origins.contains(source_origin))
    throw Failure(grpc::StatusCode::PERMISSION_DENIED, "HTTP source origin is not allowed");
  if (!cancelled) throw std::invalid_argument("HTTP cancellation callback required");
  if (!(curl_version_info(CURLVERSION_NOW)->features & CURL_VERSION_ASYNCHDNS))
    throw std::runtime_error("asynchronous libcurl DNS support is required");
  impl_->policy = std::move(policy); impl_->cancelled = std::move(cancelled);
  impl_->easy = curl_easy_init(); impl_->multi = curl_multi_init();
  if (!impl_->easy || !impl_->multi) throw std::bad_alloc();
  auto set = [&](CURLoption key, auto value) {
    if (curl_easy_setopt(impl_->easy, key, value) != CURLE_OK) throw std::runtime_error("HTTP option configuration failed");
  };
  set(CURLOPT_URL, url.c_str()); set(CURLOPT_PROTOCOLS_STR, "http,https");
  set(CURLOPT_HTTP_VERSION, static_cast<long>(CURL_HTTP_VERSION_1_1));
  set(CURLOPT_PROXY, ""); set(CURLOPT_FOLLOWLOCATION, 0L); set(CURLOPT_NOSIGNAL, 1L);
  set(CURLOPT_CONNECTTIMEOUT_MS, 10000L); set(CURLOPT_LOW_SPEED_LIMIT, 1L); set(CURLOPT_LOW_SPEED_TIME, 30L);
  set(CURLOPT_ACCEPT_ENCODING, "identity"); set(CURLOPT_HTTP_CONTENT_DECODING, 0L);
  set(CURLOPT_WRITEFUNCTION, &Impl::body); set(CURLOPT_WRITEDATA, impl_.get());
  set(CURLOPT_HEADERFUNCTION, &Impl::header); set(CURLOPT_HEADERDATA, impl_.get());
  set(CURLOPT_OPENSOCKETFUNCTION, &Impl::socket); set(CURLOPT_OPENSOCKETDATA, impl_.get());
  if (curl_multi_add_handle(impl_->multi, impl_->easy) != CURLM_OK) throw std::runtime_error("HTTP transfer initialization failed");
  impl_->added = true;
}
HttpInput::~HttpInput() = default;
size_t HttpInput::read(char* output, size_t capacity) {
  if (!output || !capacity) throw std::invalid_argument("HTTP output buffer required");
  auto& self = *impl_;
  for (;;) {
    self.check();
    if (!self.pending.empty()) {
      auto& front = self.pending.front();
      const auto n = std::min(capacity, front.size() - self.offset);
      std::memcpy(output, front.data() + self.offset, n); self.offset += n; self.bytes -= n;
      if (self.offset == front.size()) { self.pending.pop_front(); self.offset = 0; }
      // Downstream model inference is consumer work, not a stalled HTTP source.
      if (!self.finished && !self.paused) {
        if (curl_easy_pause(self.easy, CURLPAUSE_RECV) != CURLE_OK)
          throw Failure(grpc::StatusCode::UNAVAILABLE, "HTTP backpressure pause failed");
        self.paused = true;
      }
      return n;
    }
    if (self.finished) {
      if (self.result != CURLE_OK) {
        if (self.address_rejected) throw Failure(grpc::StatusCode::PERMISSION_DENIED, "HTTP destination address is not allowed");
        throw Failure(grpc::StatusCode::UNAVAILABLE, "HTTP source transfer failed");
      }
      const auto status = status_code();
      if (status < 200 || status >= 300) throw Failure(grpc::StatusCode::FAILED_PRECONDITION, "HTTP source did not return a successful response");
      return 0;
    }
    if (self.paused) {
      self.paused = false;
      if (curl_easy_pause(self.easy, CURLPAUSE_CONT) != CURLE_OK) throw Failure(grpc::StatusCode::UNAVAILABLE, "HTTP resume failed");
      self.check();
    }
    int running = 0;
    if (curl_multi_perform(self.multi, &running) != CURLM_OK) throw Failure(grpc::StatusCode::UNAVAILABLE, "HTTP polling failed");
    self.check();
    int messages = 0;
    while (auto* message = curl_multi_info_read(self.multi, &messages))
      if (message->msg == CURLMSG_DONE) { self.finished = true; self.result = message->data.result; }
    if (!self.pending.empty() || self.finished) continue;
    if (curl_multi_poll(self.multi, nullptr, 0, 100, nullptr) != CURLM_OK)
      throw Failure(grpc::StatusCode::UNAVAILABLE, "HTTP wait failed");
  }
}
size_t HttpInput::buffered_bytes() const { return impl_->bytes; }
long HttpInput::status_code() const { long code = 0; curl_easy_getinfo(impl_->easy, CURLINFO_RESPONSE_CODE, &code); return code; }
std::string HttpInput::library_version() { initialize(); return std::string("libcurl/") + curl_version_info(CURLVERSION_NOW)->version; }
}  // namespace media
