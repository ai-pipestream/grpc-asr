// RemoteTranscriptionOptions (docling ApiAsrOptions): validation, and the
// multipart form fields, which must equal docling-slim 2.140.0's own
// _ApiAsrModel._form_fields output for the same options (receipt:
// docs/receipts/feat-remote-asr-options/form-fields-reference.txt).

#include "engine/remote_options.h"

#include <limits>
#include <print>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fixture.h"

namespace asrv1 = ai::pipestream::asr::v1;
using asr::engine::remote_form_fields;
using asr::engine::remote_timeout_seconds;
using asr::engine::validate_remote_options;
using Fields = std::vector<std::pair<std::string, std::string>>;

namespace {

asrv1::RemoteTranscriptionOptions valid() {
    asrv1::RemoteTranscriptionOptions options;
    options.set_url("http://localhost:8000/v1/audio/transcriptions");
    return options;
}

std::string render(const Fields& fields) {
    std::string out;
    for (const auto& [key, value] : fields) out += key + "=" + value + "\n";
    return out;
}

void expect_fields(const asrv1::RemoteTranscriptionOptions& options, const Fields& expected,
                   const std::string& name) {
    const Fields actual = remote_form_fields(options);
    require(actual == expected, name + ": form fields differ\nactual:\n" + render(actual) +
                                    "expected:\n" + render(expected));
}

void verify_defaults_match_docling() {
    const auto options = valid();
    expect_fields(options, {{"response_format", "verbose_json"}}, "defaults");
    require(remote_timeout_seconds(options) == 600.0, "unset timeout is docling's 600 s");
}

void verify_full_options_match_docling() {
    auto options = valid();
    options.set_model("whisper-1");
    options.set_language("en");
    options.set_initial_prompt("NATO alphabet");
    options.set_word_timestamps(true);
    auto& params = *options.mutable_params();
    params["a_flag"].set_bool_value(true);
    params["b_off"].set_bool_value(false);
    params["beam"].set_int_value(5);
    params["response_format"].set_string_value("text");
    params["temperature"].set_double_value(0.0);
    params["top"].set_double_value(0.25);
    params["vad"].set_string_value("silero");
    options.set_timeout_seconds(30.5);
    expect_fields(options,
                  {{"response_format", "verbose_json"},
                   {"model", "whisper-1"},
                   {"language", "en"},
                   {"prompt", "NATO alphabet"},
                   {"timestamp_granularities[]", "segment"},
                   {"timestamp_granularities[]", "word"},
                   {"a_flag", "true"},
                   {"b_off", "false"},
                   {"beam", "5"},
                   {"temperature", "0.0"},
                   {"top", "0.25"},
                   {"vad", "silero"}},
                  "full");
    require(remote_timeout_seconds(options) == 30.5, "a set timeout is used");
}

void verify_empty_strings_are_omitted() {
    auto options = valid();
    options.set_model("");
    options.set_language("");
    options.set_initial_prompt("");
    expect_fields(options, {{"response_format", "verbose_json"}}, "empty strings omitted");
}

// Doubles print as Python's str() does: the expected strings are
// docling's own output ("floats" in form-fields-reference.txt).
void verify_floats_match_docling() {
    auto options = valid();
    auto& params = *options.mutable_params();
    params["f01"].set_double_value(1e5);
    params["f02"].set_double_value(1e15);
    params["f03"].set_double_value(1e16);
    params["f04"].set_double_value(1e-4);
    params["f05"].set_double_value(1e-5);
    params["f06"].set_double_value(123456.789);
    params["f07"].set_double_value(-0.0);
    params["f08"].set_double_value(1.5e-7);
    params["f09"].set_double_value(0.1);
    params["f10"].set_double_value(1e22);
    params["f11"].set_double_value(-2.5e-300);
    params["f12"].set_double_value(9007199254740993.0);
    params["f13"].set_double_value(5e-324);
    params["f14"].set_double_value(1.7976931348623157e308);
    expect_fields(options,
                  {{"response_format", "verbose_json"},
                   {"f01", "100000.0"},
                   {"f02", "1000000000000000.0"},
                   {"f03", "1e+16"},
                   {"f04", "0.0001"},
                   {"f05", "1e-05"},
                   {"f06", "123456.789"},
                   {"f07", "-0.0"},
                   {"f08", "1.5e-07"},
                   {"f09", "0.1"},
                   {"f10", "1e+22"},
                   {"f11", "-2.5e-300"},
                   {"f12", "9007199254740992.0"},
                   {"f13", "5e-324"},
                   {"f14", "1.7976931348623157e+308"}},
                  "floats");
}

void verify_params_keep_their_types_on_the_wire() {
    auto options = valid();
    (*options.mutable_params())["beam"].set_int_value(5);
    (*options.mutable_params())["big"].set_uint_value(18446744073709551615ULL);
    asrv1::RemoteTranscriptionOptions wire;
    require(wire.ParseFromString(options.SerializeAsString()), "parse serialized options");
    require(wire.params().at("beam").kind_case() == asrv1::ScalarValue::kIntValue,
            "an integer stays an integer");
    require(wire.params().at("big").uint_value() == 18446744073709551615ULL, "uint64 survives");
    require(asr::engine::scalar_form_value(wire.params().at("big")) == "18446744073709551615",
            "uint64 prints in full");
}

void expect_invalid(const asrv1::RemoteTranscriptionOptions& options, const std::string& needle) {
    const grpc::Status status = validate_remote_options(options);
    require(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "expected INVALID_ARGUMENT for " + needle);
    require(status.error_message().find(needle) != std::string::npos,
            "message names " + needle + ": " + status.error_message());
}

void verify_validation() {
    require(validate_remote_options(valid()).ok(), "the docling default url is valid");
    auto https = valid();
    https.set_url("HTTPS://api.example.org/v1/audio/transcriptions");
    require(validate_remote_options(https).ok(), "https, any scheme case");

    for (const char* url : {"", "file:///etc/passwd", "ftp://host/x", "localhost:8000/v1",
                            "http://", "http:///path", "http://host/a b", "http://:8000/v1",
                            "http://?q", "http://[]/v1", "http://[::1/v1", "http://host:/v1",
                            "http://host:0/v1", "http://host:65536/v1", "http://host:80x/v1"}) {
        auto options = valid();
        options.set_url(url);
        expect_invalid(options, "RemoteTranscriptionOptions.url");
    }
    for (const char* url : {"http://user:secret@host/v1", "https://token@host/v1", "http://@host/v1"}) {
        auto options = valid();
        options.set_url(url);
        expect_invalid(options, "url must not carry userinfo");
        require(validate_remote_options(options).error_message().find("secret") == std::string::npos,
                "the userinfo error does not echo the credentials");
    }
    for (const char* url : {"http://host:8000/v1", "http://[::1]:8000/v1", "http://[::1]/v1",
                            "https://host:65535", "http://host?x=a@b", "http://host/p#a@b"}) {
        auto options = valid();
        options.set_url(url);
        require(validate_remote_options(options).ok(), std::string("valid url ") + url);
    }
    auto header = valid();
    (*header.mutable_headers())["Bad Header"] = "x";
    expect_invalid(header, "headers key 'Bad Header'");
    auto injected = valid();
    (*injected.mutable_headers())["Authorization"] = "Bearer x\r\nX-Evil: 1";
    expect_invalid(injected, "headers['Authorization']");
    auto duplicate = valid();
    (*duplicate.mutable_headers())["Authorization"] = "Bearer a";
    (*duplicate.mutable_headers())["authorization"] = "Bearer b";
    expect_invalid(duplicate, "repeats another header name");
    require(validate_remote_options(duplicate).error_message().find("Bearer") == std::string::npos,
            "the duplicate-header error does not echo header values");
    auto empty_key = valid();
    (*empty_key.mutable_params())[""].set_int_value(1);
    expect_invalid(empty_key, "empty key");
    auto unset = valid();
    (void)(*unset.mutable_params())["temperature"];
    expect_invalid(unset, "params['temperature'] has no value");
    for (const double timeout : {0.0, -1.0, 3600.001, 86400.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        auto options = valid();
        options.set_timeout_seconds(timeout);
        expect_invalid(options, "timeout_seconds");
    }
    auto longest = valid();
    longest.set_timeout_seconds(3600.0);
    require(validate_remote_options(longest).ok(), "a timeout of exactly 3600 s is allowed");
}

}  // namespace

int main() {
    try {
        verify_defaults_match_docling();
        verify_full_options_match_docling();
        verify_empty_strings_are_omitted();
        verify_floats_match_docling();
        verify_params_keep_their_types_on_the_wire();
        verify_validation();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("remote-options-test passed");
    return 0;
}
