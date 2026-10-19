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
                            "http://", "http:///path", "http://host/a b"}) {
        auto options = valid();
        options.set_url(url);
        expect_invalid(options, "url");
    }
    auto header = valid();
    (*header.mutable_headers())["Bad Header"] = "x";
    expect_invalid(header, "headers key 'Bad Header'");
    auto injected = valid();
    (*injected.mutable_headers())["Authorization"] = "Bearer x\r\nX-Evil: 1";
    expect_invalid(injected, "headers['Authorization']");
    auto empty_key = valid();
    (*empty_key.mutable_params())[""].set_int_value(1);
    expect_invalid(empty_key, "empty key");
    auto unset = valid();
    (void)(*unset.mutable_params())["temperature"];
    expect_invalid(unset, "params['temperature'] has no value");
    for (const double timeout : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        auto options = valid();
        options.set_timeout_seconds(timeout);
        expect_invalid(options, "timeout_seconds");
    }
}

}  // namespace

int main() {
    try {
        verify_defaults_match_docling();
        verify_full_options_match_docling();
        verify_empty_strings_are_omitted();
        verify_params_keep_their_types_on_the_wire();
        verify_validation();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("remote-options-test passed");
    return 0;
}
