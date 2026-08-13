#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "config.h"
#include "engine/model_pool.h"
#include "service/asr_service.h"
#include "whisper.h"

namespace {

// The handler only writes one byte to a pipe: grpc::Server::Shutdown
// allocates and takes locks, none of which is async-signal-safe. A
// dedicated thread does the real shutdown.
int g_shutdown_pipe[2] = {-1, -1};

void handle_shutdown(int /*signum*/) {
    char one = 1;
    [[maybe_unused]] ssize_t n = ::write(g_shutdown_pipe[1], &one, 1);
}

void install_shutdown_pipe() {
    if (::pipe2(g_shutdown_pipe, O_CLOEXEC) != 0) {
        throw std::runtime_error("shutdown pipe creation failed");
    }
    struct sigaction action = {};
    action.sa_handler = handle_shutdown;
    ::sigaction(SIGINT, &action, nullptr);
    ::sigaction(SIGTERM, &action, nullptr);
    ::signal(SIGPIPE, SIG_IGN);
}

// whisper logs to stderr by default; keep model-load noise but drop the
// per-inference chatter.
void quiet_whisper_log(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::cerr << text;
    }
}

}  // namespace

int main() {
    try {
        whisper_log_set(quiet_whisper_log, nullptr);
        install_shutdown_pipe();

        const asr::Config config = asr::load_config_from_env();

        // The pool verifies the backend and loads every model; a missing
        // backend or weight file stops the process here, not on the first
        // RPC.
        asr::engine::ModelPool pool(config);
        asr::AsrServiceImpl service(config, pool);

        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        grpc::ServerBuilder builder;
        builder.AddListeningPort(config.listen_address, grpc::InsecureServerCredentials());
        builder.SetMaxReceiveMessageSize(4 * 1024 * 1024);
        builder.RegisterService(&service);
        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
        if (server == nullptr) {
            std::cerr << "Unable to listen on " << config.listen_address << '\n';
            return 1;
        }

        std::thread shutdown_thread([&server] {
            char byte = 0;
            [[maybe_unused]] ssize_t n = ::read(g_shutdown_pipe[0], &byte, 1);
            server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(10));
        });

        // Interval metrics line; a condition variable instead of a sleep
        // loop so shutdown never waits out the interval.
        std::mutex metrics_mutex;
        std::condition_variable metrics_stop;
        bool stopping = false;
        std::thread metrics_thread;
        if (config.metrics_interval_seconds > 0) {
            metrics_thread = std::thread([&] {
                std::unique_lock<std::mutex> lock(metrics_mutex);
                while (!metrics_stop.wait_for(
                    lock, std::chrono::seconds(config.metrics_interval_seconds),
                    [&] { return stopping; })) {
                    std::cout << "grpc-asr metrics: media{transcribed="
                              << service.transcribed.load()
                              << ",rejected=" << service.rejected.load()
                              << ",failed=" << service.failed.load()
                              << "} audio_seconds=" << service.audio_ms.load() / 1000
                              << std::endl;
                }
            });
        }

        std::cout << "grpc-asr listening on " << config.listen_address << " (whisper.cpp "
                  << whisper_version() << ", backend " << pool.backend() << ")" << std::endl;
        server->Wait();

        // Wake the shutdown thread if Wait() returned for another reason.
        handle_shutdown(0);
        shutdown_thread.join();
        {
            std::lock_guard<std::mutex> lock(metrics_mutex);
            stopping = true;
        }
        metrics_stop.notify_all();
        if (metrics_thread.joinable()) {
            metrics_thread.join();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Startup failed: " << error.what() << '\n';
        return 1;
    }
}
