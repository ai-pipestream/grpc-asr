#!/usr/bin/env bash
# Builds the ffmpeg and ffprobe every grpc-asr image ships, from a pinned
# upstream release, LGPL only: no --enable-gpl, no --enable-nonfree, no
# external libraries but zlib (whose license is permissive). The result
# installs under PREFIX (default /opt/ffmpeg) as two executables and the
# shared libraries they load through an absolute rpath, so the runtime
# stage copies one directory and nothing else.
#
# Only what the service runs is compiled in (see src/media/video_demux.cpp):
#   - ffprobe over the media: format duration, first audio and video stream
#   - ffmpeg -map a:0 -f f32le -ac 1 -ar 16000 pipe:1   (audio for whisper)
#   - ffmpeg -map v:0 -vf fps=1/N -f image2pipe -c:v png pipe:1   (stills)
# over the containers that take the ffmpeg path (Ogg, MP4/M4A/MOV,
# Matroska/WebM) plus the audio containers the in-process decoder also
# handles (WAV, MP3, FLAC, raw ADTS AAC), read from /dev/fd/3 (file
# protocol) and written to pipe:1. Anything else fails as undecodable.
#
# Corresponding source: the verified tarball is copied next to the license
# texts and the exact configure line under PREFIX/share/doc/ffmpeg, so the
# image carries what LGPL-2.1 section 4 asks for.
set -euo pipefail

FFMPEG_VERSION=9.0.2
FFMPEG_SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
FFMPEG_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"

prefix=${1:-/opt/ffmpeg}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

tarball="$work/ffmpeg-${FFMPEG_VERSION}.tar.xz"
curl -fsSL --retry 3 -o "$tarball" "$FFMPEG_URL"
echo "${FFMPEG_SHA256}  ${tarball}" | sha256sum -c -
tar -xJf "$tarball" -C "$work"
cd "$work/ffmpeg-${FFMPEG_VERSION}"

# CC picks the compiler (the CUDA image builds with gcc-14).
# --disable-autodetect keeps configure from picking up whatever the build
# image happens to carry (CUDA headers, VA-API, X11); threads and zlib are
# then the only things switched on by name (iconv, which glibc carries,
# is off too: nothing here converts subtitle charsets).
configure_args=(
  --prefix="$prefix" --cc="${CC:-gcc}"
  --enable-shared --disable-static --enable-rpath
  --disable-autodetect --enable-pthreads --enable-zlib --disable-iconv
  --disable-everything --disable-doc --disable-ffplay --disable-avdevice
  --disable-network --disable-debug
  --enable-protocol=file,pipe
  --enable-demuxer=ogg,mov,matroska,wav,mp3,flac,aac
  --enable-decoder=aac,aac_fixed,aac_latm,mp3,mp3float,mp2,mp2float,flac,alac,vorbis,opus,ac3,eac3,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s32le,pcm_f32le,pcm_u8,pcm_alaw,pcm_mulaw
  --enable-decoder=h264,hevc,vp8,vp9,mpeg4,mpeg1video,mpeg2video,mjpeg,h263,theora,prores
  --enable-parser=aac,aac_latm,mpegaudio,flac,vorbis,opus,ac3,h264,hevc,vp8,vp9,mpeg4video,mpegvideo,mjpeg,h263
  --enable-encoder=pcm_f32le,png
  --enable-muxer=pcm_f32le,image2pipe
  --enable-filter=aresample,aformat,format,scale,fps,null,anull
)
./configure "${configure_args[@]}"

# configure prints the license it settled on; refuse anything but LGPL.
license=$(grep -E '^#define FFMPEG_LICENSE ' config.h)
if [[ "$license" != *'"LGPL version 2.1 or later"'* ]]; then
  echo "ffmpeg configured as $license, expected LGPL version 2.1 or later" >&2
  exit 1
fi

make -j"$(nproc)"
make install

doc="$prefix/share/doc/ffmpeg"
mkdir -p "$doc"
cp COPYING.LGPLv2.1 LICENSE.md "$doc/"
cp "$tarball" "$doc/"
printf '%s\n' "./configure ${configure_args[*]}" > "$doc/CONFIGURE"
# Headers, pkg-config files and the share/ffmpeg presets are not needed
# at run time.
rm -rf "$prefix/include" "$prefix/lib/pkgconfig" "$prefix/share/ffmpeg"

"$prefix/bin/ffmpeg" -hide_banner -buildconf
