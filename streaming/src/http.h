#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>

namespace media {
struct HttpPolicy {
  // Exact canonical origins: scheme://host:port. Empty denies all network input.
  std::set<std::string> allowed_origins;
  // Explicit operator opt-in for feeds on private/local networks.
  bool allow_private_addresses = false;
};
class HttpInput {
 public:
  HttpInput(std::string url, HttpPolicy policy, std::function<bool()> cancelled);
  ~HttpInput();
  // Returns response entity bytes, zero at successful EOF, or throws Failure.
  // Transfer framing is removed; content decompression is disabled.
  size_t read(char* output, size_t capacity);
  size_t buffered_bytes() const;
  long status_code() const;
  static std::string origin(const std::string& url);
  static std::string library_version();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace media
