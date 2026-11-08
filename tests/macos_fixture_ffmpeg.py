#!/usr/bin/env python3
"""Test-only fixture generator for Homebrew FFmpeg without libvorbis.

Selected explicitly through GRPC_ASR_TEST_FIXTURE_FFMPEG. Native Vorbis
requires stereo; the decoder under test must still identify Vorbis and
produce the expected transcript. This does not run service media decoding.
"""
import os
import sys

args = sys.argv[1:]
if "libvorbis" in args:
    args[args.index("libvorbis")] = "vorbis"
    args[-1:-1] = ["-strict", "experimental", "-ac", "2"]
if args and args[-1].endswith((".mp4", ".ogg")):
    args[-1:-1] = ["-threads", "2"]
os.execv("/opt/homebrew/bin/ffmpeg", ["ffmpeg", "-filter_threads", "2",
    "-filter_complex_threads", "2", *args])
