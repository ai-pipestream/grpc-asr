#!/usr/bin/env python3
"""Retain a real tiny.en/JFK RPC transcript; complements the C++ lifecycle suite."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'ai.pipestream.asr.v1.AsrService/'


def events(raw):
    decoder = json.JSONDecoder()
    while raw.strip():
        value, end = decoder.raw_decode(raw.lstrip())
        raw = raw.lstrip()[end:]
        yield value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audio = args.sample.read_bytes()
    payload = '\n'.join([json.dumps({'options': {'model': 'tiny.en', 'language': 'en',
        'wordTimestamps': True, 'emitDocument': True, 'filename': 'jfk.wav'}}),
        json.dumps({'chunk': {'data': base64.b64encode(audio).decode()}})])
    timings = {}
    for method, data in [('GetServiceInfo', '{}'), ('Transcribe', payload)]:
        command = ['grpcurl', '-plaintext', '-max-time', '180', '-import-path', str(ROOT / 'proto'),
            '-proto', 'ai/pipestream/asr/v1/asr_service.proto', '-d', '@', args.address, SERVICE + method]
        start = time.monotonic()
        response = subprocess.run(command, input=data, text=True, capture_output=True, timeout=190)
        timings[method] = time.monotonic() - start
        (args.output / (method + '.stdout')).write_text(response.stdout)
        (args.output / (method + '.stderr')).write_text(response.stderr)
        if response.returncode:
            raise RuntimeError(f'{method} failed: {response.stderr}')
    result = list(events((args.output / 'Transcribe.stdout').read_text()))
    assert result and 'mediaInfo' in result[0] and 'complete' in result[-1]
    segments = [item['finalSegment'] for item in result if 'finalSegment' in item]
    text = ' '.join(item['text'] for item in segments)
    normalized = ' '.join(text.lower().split())
    assert 'country can do for you' in normalized, text
    assert 'you can do for your country' in normalized, text
    previous = 0
    for index, segment in enumerate(segments):
        start, end = int(segment.get('startMs', 0)), int(segment['endMs'])
        assert int(segment.get('index', 0)) == index
        assert previous <= start < end <= 12000, segment
        previous = end
    assert any('document' in item for item in result)
    assert int(result[-1]['complete']['segmentCount']) == len(segments)
    assert result[-1]['complete']['language'] == 'en'
    receipt = {'fixture_sha256': hashlib.sha256(audio).hexdigest(),
        'seconds': timings, 'text': text, 'segments': segments,
        'scope': 'actual RPC content/timestamps; device placement and lifecycle checked separately'}
    (args.output / 'assertions.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('PASS actual RPC JFK text, ordered bounded timestamps, document and completion: ' + text)


if __name__ == '__main__':
    main()
