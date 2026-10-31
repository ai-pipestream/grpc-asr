#include "decoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libavutil/mem.h>
#include <libswresample/swresample.h>
}

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace media {
namespace {
constexpr size_t kEventBytes = 256 * 1024;
constexpr size_t kPacketBytes = 16 * 1024 * 1024;
constexpr int kOutputRate = 16000;
constexpr AVRational kMicros{1, 1000000};

void limit(bool ok, const char* message) {
  if (!ok) throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, message);
}
std::string version(unsigned int value) {
  return std::to_string(value >> 16) + "." + std::to_string((value >> 8) & 255) + "." + std::to_string(value & 255);
}
void ratio(mediapb::Rational* result, AVRational value) {
  result->set_numerator(value.num);
  result->set_denominator(value.den);
}
void metadata(google::protobuf::RepeatedPtrField<mediapb::NativeMetadata>* out, AVDictionary* values) {
  const AVDictionaryEntry* entry = nullptr;
  size_t total = 0;
  while ((entry = av_dict_get(values, "", entry, AV_DICT_IGNORE_SUFFIX))) {
    const auto key_size = strnlen(entry->key, kEventBytes + 1);
    const auto value_size = strnlen(entry->value, kEventBytes + 1);
    total += key_size + value_size;
    limit(total <= kEventBytes && out->size() < 256, "native metadata exceeds limit");
    auto* item = out->Add();
    item->set_key(entry->key, key_size);
    item->set_value(entry->value, value_size);
  }
}
template<class Parameters, class Stream>
void track_side_data(mediapb::TrackInfo* out, const Parameters* parameters, const Stream* stream) {
  const AVPacketSideData* entries;
  int count;
  // FFmpeg moved track side data into codec parameters. Feature detection
  // avoids using deprecated stream fields when the new representation exists.
  if constexpr (requires { parameters->coded_side_data; }) {
    entries = parameters->coded_side_data; count = parameters->nb_coded_side_data;
  } else {
    entries = stream->side_data; count = stream->nb_side_data;
  }
  limit(count >= 0 && count <= 128, "too many track side-data entries");
  size_t total = 0;
  for (int i = 0; i < count; ++i) {
    const auto& side = entries[i];
    limit(side.size <= kEventBytes / 2 - total, "track side data exceeds limit");
    total += side.size;
    auto* data = out->add_side_data();
    data->set_native_type(side.type);
    if (const auto* name = av_packet_side_data_name(side.type)) data->set_type_name(name);
    if (side.size) data->set_data(side.data, side.size);
  }
}
mediapb::TrackKind kind(AVMediaType value) {
  switch (value) {
    case AVMEDIA_TYPE_AUDIO: return mediapb::TRACK_KIND_AUDIO;
    case AVMEDIA_TYPE_VIDEO: return mediapb::TRACK_KIND_VIDEO;
    case AVMEDIA_TYPE_SUBTITLE: return mediapb::TRACK_KIND_SUBTITLE;
    case AVMEDIA_TYPE_DATA: return mediapb::TRACK_KIND_DATA;
    case AVMEDIA_TYPE_ATTACHMENT: return mediapb::TRACK_KIND_ATTACHMENT;
    default: return mediapb::TRACK_KIND_UNSPECIFIED;
  }
}

struct AudioState {
  AVCodecContext* codec = nullptr;
  SwrContext* resampler = nullptr;
  AVChannelLayout input_layout{};
  int input_rate = 0;
  AVSampleFormat input_format = AV_SAMPLE_FMT_NONE;
  uint64_t samples = 0;
  uint64_t input_samples = 0;
  uint64_t chunks = 0;
  uint64_t origin_sample = 0;
  std::optional<int64_t> origin_us;
  std::optional<int64_t> next_input_us;
  bool discontinuity = false;
  ~AudioState() {
    swr_free(&resampler);
    av_channel_layout_uninit(&input_layout);
    avcodec_free_context(&codec);
  }
};

class Decoder {
 public:
  Decoder(ReadMedia read, EmitMedia emit, std::function<bool()> cancelled, const mediapb::DecodeOptions& options)
      : read_(std::move(read)), emit_(std::move(emit)), cancelled_(std::move(cancelled)), options_(options),
        selected_(options.audio_track_ids().begin(), options.audio_track_ids().end()) {}
  ~Decoder() {
    av_packet_free(&packet_);
    av_frame_free(&frame_);
    avformat_close_input(&format_);
    if (io_) av_freep(&io_->buffer);
    avio_context_free(&io_);
  }
  void run() {
    auto* buffer = static_cast<unsigned char*>(av_malloc(4096));
    if (!buffer) throw std::bad_alloc();
    io_ = avio_alloc_context(buffer, 4096, 0, this, &read_callback, nullptr, nullptr);
    if (!io_) { av_free(buffer); throw std::bad_alloc(); }
    io_->seekable = 0;
    format_ = avformat_alloc_context();
    if (!format_) throw std::bad_alloc();
    format_->pb = io_;
    format_->flags |= AVFMT_FLAG_CUSTOM_IO | AVFMT_FLAG_IGNIDX;
    format_->probesize = 65536;
    format_->max_analyze_duration = 0;
    format_->max_index_size = 1024 * 1024;
    format_->max_streams = 64;
    format_->interrupt_callback = {&interrupt_callback, this};
    // Demuxers must never open filesystem paths or nested network URLs.
    format_->io_open = [](AVFormatContext*, AVIOContext**, const char*, int, AVDictionary**) { return AVERROR(EPERM); };
    AVDictionary* options = nullptr;
    av_dict_set(&options, "format_whitelist", "wav,mp3,flac,ogg,mov,matroska,webm,mpegts,aac,nut", 0);
    const int opened = avformat_open_input(&format_, nullptr, nullptr, &options);
    av_dict_free(&options);
    check(opened, "open media");
    // The PCM demuxer's default can accumulate seconds of mono speech before
    // returning a packet. Bound it explicitly for interactive streaming.
    if (std::string_view(format_->iformat->name) == "wav")
      check(av_opt_set_int(format_->priv_data, "max_size", 4096, 0), "set WAV packet size");
    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !packet_) throw std::bad_alloc();
    describe();
    while (true) {
      check_cancelled();
      const int result = av_read_frame(format_, packet_);
      if (reader_error_) std::rethrow_exception(reader_error_);
      if (result == AVERROR_EOF) break;
      check(result, "read packet");
      limit(packet_->size >= 0 && static_cast<size_t>(packet_->size) <= kPacketBytes, "native packet exceeds limit");
      if (packet_->stream_index < 0 || static_cast<unsigned>(packet_->stream_index) >= format_->nb_streams)
        throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "packet has invalid track index");
      describe();
      if (options_.emit_packets()) emit_packet();
      auto* track = format_->streams[packet_->stream_index];
      if (options_.emit_audio() && track->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
          (selected_.empty() || selected_.contains(track->index))) {
        if (packet_->flags & AV_PKT_FLAG_CORRUPT)
          throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "corrupt audio packet");
        auto& state = audio(track);
        int sent = avcodec_send_packet(state.codec, packet_);
        if (sent == AVERROR(EAGAIN)) { receive(track, state); sent = avcodec_send_packet(state.codec, packet_); }
        check(sent, "submit audio packet");
        receive(track, state);
      }
      av_packet_unref(packet_);
    }
    check_cancelled();
    for (const auto& [id, state] : audio_) {
      check(avcodec_send_packet(state->codec, nullptr), "flush audio decoder");
      receive(format_->streams[id], *state);
      flush_resampler(id, *state);
      const auto* track = format_->streams[id];
      if (std::string_view(format_->iformat->name) == "wav" && track->duration > 0 &&
          state->input_rate > 0 && std::string_view(avcodec_get_name(track->codecpar->codec_id)).starts_with("pcm_")) {
        const auto expected = av_rescale_q(track->duration, track->time_base, {1, state->input_rate});
        if (expected > 0 && state->input_samples < static_cast<uint64_t>(expected))
          throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "truncated PCM WAV data");
      }
    }
    for (auto id : selected_) if (!audio_.contains(id))
      throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "selected audio track absent or empty");
    describe();
  }

 private:
  static int read_callback(void* opaque, uint8_t* bytes, int size) noexcept {
    auto& self = *static_cast<Decoder*>(opaque);
    try {
      self.check_cancelled();
      const int read = self.read_(bytes, size);
      if (read < 0 || read > size) throw std::logic_error("invalid source callback byte count");
      return read == 0 ? AVERROR_EOF : read;
    } catch (...) {
      self.reader_error_ = std::current_exception();
      return AVERROR_EXTERNAL;
    }
  }
  static int interrupt_callback(void* opaque) noexcept {
    try { return static_cast<Decoder*>(opaque)->cancelled_(); } catch (...) { return 1; }
  }
  void check_cancelled() {
    if (cancelled_()) throw Failure(grpc::StatusCode::CANCELLED, "media decode cancelled");
  }
  void check(int result, const char* operation) {
    if (reader_error_) std::rethrow_exception(reader_error_);
    check_cancelled();
    if (result >= 0) return;
    std::array<char, AV_ERROR_MAX_STRING_SIZE> error{};
    av_strerror(result, error.data(), error.size());
    throw Failure(result == AVERROR(ENOMEM) ? grpc::StatusCode::RESOURCE_EXHAUSTED : grpc::StatusCode::INVALID_ARGUMENT,
                  std::string(operation) + ": " + error.data());
  }
  void describe() {
    mediapb::StreamEvent event;
    auto* info = event.mutable_media_info();
    info->set_container_name(format_->iformat->name);
    if (format_->iformat->long_name) info->set_container_description(format_->iformat->long_name);
    if (format_->start_time != AV_NOPTS_VALUE) info->set_start_time_us(format_->start_time);
    if (format_->duration != AV_NOPTS_VALUE) info->set_duration_us(format_->duration);
    metadata(info->mutable_metadata(), format_->metadata);
    limit(format_->nb_streams <= 64, "too many media tracks");
    for (unsigned i = 0; i < format_->nb_streams; ++i) {
      const auto* track = format_->streams[i];
      const auto* parameters = track->codecpar;
      auto* out = info->add_tracks();
      out->set_track_id(i);
      out->set_native_id(track->id);
      out->set_kind(kind(parameters->codec_type));
      out->set_native_type(parameters->codec_type);
      out->set_codec_id(parameters->codec_id);
      out->set_codec_name(avcodec_get_name(parameters->codec_id));
      out->set_codec_tag(parameters->codec_tag);
      limit(parameters->extradata_size >= 0 && static_cast<size_t>(parameters->extradata_size) <= kEventBytes, "codec initialization exceeds limit");
      if (parameters->extradata_size) out->set_codec_extradata(parameters->extradata, parameters->extradata_size);
      ratio(out->mutable_time_base(), track->time_base);
      if (track->start_time != AV_NOPTS_VALUE) out->set_start_time(track->start_time);
      if (track->duration != AV_NOPTS_VALUE) out->set_duration(track->duration);
      metadata(out->mutable_metadata(), track->metadata);
      out->set_sample_rate(std::max(0, parameters->sample_rate));
      out->set_channels(std::max(0, parameters->ch_layout.nb_channels));
      out->set_width(std::max(0, parameters->width));
      out->set_height(std::max(0, parameters->height));
      out->set_disposition_flags(track->disposition);
      out->set_bit_rate(parameters->bit_rate);
      out->set_profile(parameters->profile);
      out->set_level(parameters->level);
      track_side_data(out, parameters, track);
      limit(info->ByteSizeLong() <= kEventBytes, "media description exceeds limit");
    }
    const auto serialized = info->SerializeAsString();
    if (serialized != last_description_) {
      last_description_ = serialized;
      emit_(std::move(event));
    }
  }
  void emit_packet() {
    if (packets_ == std::numeric_limits<uint64_t>::max())
      throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "packet counter exhausted");
    ++packets_;
    const auto* track = format_->streams[packet_->stream_index];
    size_t offset = 0;
    do {
      mediapb::StreamEvent event;
      auto* out = event.mutable_packet();
      out->set_packet_number(packets_);
      out->set_track_id(track->index);
      out->set_fragment_offset(offset);
      out->set_packet_bytes(packet_->size);
      if (packet_->pts != AV_NOPTS_VALUE) out->set_pts(packet_->pts);
      if (packet_->dts != AV_NOPTS_VALUE) out->set_dts(packet_->dts);
      out->set_duration(packet_->duration);
      ratio(out->mutable_time_base(), track->time_base);
      out->set_native_flags(packet_->flags);
      out->set_keyframe(packet_->flags & AV_PKT_FLAG_KEY);
      if (packet_->pos >= 0) out->set_source_byte_position(packet_->pos);
      if (!offset) {
        limit(packet_->side_data_elems >= 0 && packet_->side_data_elems <= 128, "too many side-data entries");
        for (int i = 0; i < packet_->side_data_elems; ++i) {
          const auto& side = packet_->side_data[i];
          limit(side.size <= kEventBytes / 2, "packet side data exceeds limit");
          auto* data = out->add_side_data();
          data->set_native_type(side.type);
          if (const auto* name = av_packet_side_data_name(side.type)) data->set_type_name(name);
          if (side.size) data->set_data(side.data, side.size);
          limit(out->ByteSizeLong() < kEventBytes / 2, "packet side data exceeds limit");
        }
      }
      const auto count = std::min(static_cast<size_t>(packet_->size) - offset, kEventBytes / 2);
      if (count) out->set_data(packet_->data + offset, count);
      emit_(std::move(event));
      offset += count;
    } while (offset < static_cast<size_t>(packet_->size));
  }
  AudioState& audio(const AVStream* track) {
    auto& ptr = audio_[track->index];
    if (ptr) return *ptr;
    ptr = std::make_unique<AudioState>();
    auto* decoder = avcodec_find_decoder(track->codecpar->codec_id);
    if (!decoder) throw Failure(grpc::StatusCode::UNIMPLEMENTED, "audio codec unavailable in this build");
    ptr->codec = avcodec_alloc_context3(decoder);
    if (!ptr->codec) throw std::bad_alloc();
    check(avcodec_parameters_to_context(ptr->codec, track->codecpar), "copy audio codec parameters");
    ptr->codec->thread_count = 1;
    ptr->codec->max_samples = 1024 * 1024;
    ptr->codec->pkt_timebase = track->time_base;
    ptr->codec->err_recognition = AV_EF_CAREFUL;
    check(avcodec_open2(ptr->codec, decoder, nullptr), "open audio decoder");
    return *ptr;
  }
  void receive(const AVStream* track, AudioState& state) {
    while (true) {
      check_cancelled();
      const int result = avcodec_receive_frame(state.codec, frame_);
      if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
      check(result, "decode audio frame");
      convert(track, state);
      av_frame_unref(frame_);
    }
  }
  void convert(const AVStream* track, AudioState& state) {
    limit(frame_->sample_rate > 0 && frame_->sample_rate <= 384000 && frame_->ch_layout.nb_channels > 0 &&
          frame_->ch_layout.nb_channels <= 32 && frame_->nb_samples >= 0 && frame_->nb_samples <= 1024 * 1024,
          "decoded audio dimensions exceed limit");
    limit(state.input_samples <= UINT64_MAX - frame_->nb_samples, "input sample counter exhausted");
    state.input_samples += frame_->nb_samples;
    if (!state.resampler) {
      state.input_rate = frame_->sample_rate;
      state.input_format = static_cast<AVSampleFormat>(frame_->format);
      check(av_channel_layout_copy(&state.input_layout, &frame_->ch_layout), "copy channel layout");
      AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
      check(swr_alloc_set_opts2(&state.resampler, &mono, AV_SAMPLE_FMT_FLT, kOutputRate,
                               &frame_->ch_layout, state.input_format, state.input_rate, 0, nullptr), "allocate audio resampler");
      check(swr_init(state.resampler), "initialize audio resampler");
    } else if (state.input_rate != frame_->sample_rate || state.input_format != frame_->format ||
               av_channel_layout_compare(&state.input_layout, &frame_->ch_layout) != 0) {
      throw Failure(grpc::StatusCode::UNIMPLEMENTED, "midstream audio format change requires a new stream");
    }
    if (frame_->best_effort_timestamp != AV_NOPTS_VALUE) {
      const auto timestamp = av_rescale_q(frame_->best_effort_timestamp, track->time_base, kMicros);
      const bool discontinuity = state.next_input_us &&
          (static_cast<long double>(timestamp) - *state.next_input_us > 2000 ||
           static_cast<long double>(timestamp) - *state.next_input_us < -2000);
      if (discontinuity) {
        flush_resampler(track->index, state);
        swr_close(state.resampler);
        check(swr_init(state.resampler), "reset audio resampler");
        state.discontinuity = true;
      }
      if (!state.origin_us || discontinuity) {
        state.origin_us = timestamp;
        state.origin_sample = state.samples;
      }
      state.next_input_us = av_sat_add64(timestamp, av_rescale_q(frame_->nb_samples, {1, state.input_rate}, kMicros));
    } else if (state.next_input_us) {
      state.next_input_us = av_sat_add64(*state.next_input_us, av_rescale_q(frame_->nb_samples, {1, state.input_rate}, kMicros));
    }
    const int capacity = swr_get_out_samples(state.resampler, frame_->nb_samples);
    check(capacity, "size resampled audio");
    limit(capacity <= 1024 * 1024, "resampled audio exceeds limit");
    std::vector<float> samples(capacity);
    auto* output = reinterpret_cast<uint8_t*>(samples.data());
    const int count = swr_convert(state.resampler, &output, capacity,
                                 const_cast<const uint8_t**>(frame_->extended_data), frame_->nb_samples);
    check(count, "resample audio");
    emit_audio(track->index, state, samples.data(), count);
  }
  void flush_resampler(unsigned track, AudioState& state) {
    if (!state.resampler) return;
    std::array<float, 4096> samples{};
    while (true) {
      auto* output = reinterpret_cast<uint8_t*>(samples.data());
      const int count = swr_convert(state.resampler, &output, samples.size(), nullptr, 0);
      check(count, "flush audio resampler");
      if (!count) break;
      emit_audio(track, state, samples.data(), count);
    }
  }
  void emit_audio(unsigned track, AudioState& state, const float* samples, size_t count) {
    for (size_t offset = 0; offset < count;) {
      const auto length = std::min(size_t{4096}, count - offset);
      limit(state.samples <= static_cast<uint64_t>(INT64_MAX) - length && state.chunks < UINT64_MAX, "audio timeline exhausted");
      mediapb::StreamEvent event;
      auto* out = event.mutable_audio();
      out->set_chunk_number(++state.chunks);
      out->set_track_id(track);
      out->set_first_sample(state.samples);
      out->set_sample_rate(kOutputRate);
      out->set_channels(1);
      out->set_discontinuity(state.discontinuity);
      state.discontinuity = false;
      if (state.origin_us)
        out->set_presentation_time_us(av_sat_add64(*state.origin_us, av_rescale_q(state.samples - state.origin_sample, {1, kOutputRate}, kMicros)));
      std::string bytes(length * 4, '\0');
      for (size_t i = 0; i < length; ++i) {
        const auto bits = std::bit_cast<uint32_t>(samples[offset + i]);
        for (int byte = 0; byte < 4; ++byte) bytes[i * 4 + byte] = static_cast<char>((bits >> (byte * 8)) & 255);
      }
      out->set_pcm_f32le(std::move(bytes));
      state.samples += length;
      offset += length;
      emit_(std::move(event));
    }
  }
  ReadMedia read_;
  EmitMedia emit_;
  std::function<bool()> cancelled_;
  mediapb::DecodeOptions options_;
  std::set<unsigned> selected_;
  AVFormatContext* format_ = nullptr;
  AVIOContext* io_ = nullptr;
  AVPacket* packet_ = nullptr;
  AVFrame* frame_ = nullptr;
  std::exception_ptr reader_error_;
  std::map<unsigned, std::unique_ptr<AudioState>> audio_;
  uint64_t packets_ = 0;
  std::string last_description_;
};
}  // namespace

std::string decoder_version() {
  return "libavformat/" + version(avformat_version()) + " libavcodec/" + version(avcodec_version()) +
         " libavutil/" + version(avutil_version()) + " libswresample/" + version(swresample_version());
}
void decode_media(ReadMedia read, EmitMedia emit, std::function<bool()> cancelled, const mediapb::DecodeOptions& options) {
  // A single malicious native allocation must not reserve arbitrary process
  // memory. This is process-wide because FFmpeg exposes a process-wide limit.
  static const bool bounded_allocations = [] { av_max_alloc(32 * 1024 * 1024); return true; }();
  (void)bounded_allocations;
  Decoder decoder(std::move(read), std::move(emit), std::move(cancelled), options);
  decoder.run();
}
}  // namespace media
