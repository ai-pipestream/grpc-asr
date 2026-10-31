# Speech translation fixtures

## Human recording: MINDS-14

`minds14-es-0.wav` is Spanish audio from PolyAI's MINDS-14 dataset:
configuration `es-ES`, split `train`, row 0, original path
`es-ES~APP_ERROR/60266571dc3604f481bcb50a.wav`.

Source and attribution: PolyAI, Daniela Gerz et al., *Multilingual and
Cross-Lingual Intent Detection from Spoken Data* (2021).
Dataset revision: `40ce77cb32a384e4d50a568e1ec39ac804019d33`.
[Dataset and license declaration](https://huggingface.co/datasets/PolyAI/minds14/blob/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md).
This audio is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
separately from the project code. It is the WAV supplied by the dataset viewer,
with no additional audio modification here. SHA-256:
`8a350c47afd69d9fc7f01f6a7bdfa4ccec3515e2d0bdef218e7e2c17468f99b4`.

The published Spanish transcript describes a problem with the application
while trying to make a bank transfer to a known account. The test requires
those three facts (application, transfer, account) in both Spanish transcription
and English translation, correct source/output languages, model provenance,
native tokens and stream completion. This one recording is a functional gate,
not a representative accuracy benchmark.

## Synthetic challenge

`spanish.txt` is project-authored text. `spanish.wav` is synthetic Spanish
speech generated from that text with eSpeak NG 1.52.0, Ubuntu package
`1.52.0+dfsg-5build1`, Spanish voice `es`, speed 135 words per minute:

```sh
espeak-ng -v es -s 135 -w spanish.wav -f spanish.txt
```

The WAV SHA-256 is
`21211f165720ad3af90b62362a2e5f737639ad1d6d94d85dfe32d48ed9935e42`.
The checked-in audio is used directly by tests. eSpeak is only a fixture
creation tool, not a build, deployment or request-processing dependency.

Meaning: the speaker greets the listener, says her name is María, lives in
Madrid, works in a library, and enjoys reading books and learning languages.
The translation test checks Spanish source-language metadata, distinct
Spanish transcription and English translation, the Madrid and library facts,
model identity, native tokens, and explicit stream completion. This is a
synthetic functional test, not a human-speech accuracy benchmark.

The first candidate model asset was multilingual `ggml-tiny.bin`, from
`https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin`,
SHA-256 `be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21`.
Weights are not committed here and are never fetched by requests.

The tiny model failed this fixture: it detected Spanish but its English output
lost Madrid and the library meaning. Do not treat a successful RPC or correct
language labels as evidence of accurate translation.

The base model (`ggml-base.bin`) also failed the synthetic case, misrecognizing
`biblioteca` in the Spanish transcription. Keep the strict challenge available:

```sh
./build/streaming/translation_tests /path/to/multilingual-model.bin streaming/tests/fixtures/spanish.wav synthetic
```

The human-recording gate uses `MEDIA_TEST_MULTILINGUAL_MODEL` and
`ctest --test-dir build -R translation_tests --output-on-failure`. The base
candidate is pinned to the same model repository revision above, filename
`ggml-base.bin`, SHA-256
`60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe`.

The base model passed the human-recording functional gate over real gRPC,
including Spanish auto-detection and separate transcription/translation tasks.
Its English output retained application, bank transfer and account, but rendered
“known account” awkwardly as “knowledge account”. This gate does not measure
full sentence accuracy and must not be presented as production translation
quality. Broader human-recording and model qualification remains required.
