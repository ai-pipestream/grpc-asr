#pragma once
#include <string>
#include <vector>
#include <string_view>

std::string wav_fixture(unsigned sample_rate = 16000, unsigned channels = 1, unsigned seconds = 3);
std::string nut_fixture(unsigned width = 4, unsigned height = 4, unsigned frames = 30);

struct VideoFixture {
  std::string bytes;
  std::vector<std::string> video_packets;
};
VideoFixture matroska_fixture(std::string_view pcm16 = {});
std::string fragmented_mp4_fixture(std::string_view pcm16 = {}, int rotation = 0);
