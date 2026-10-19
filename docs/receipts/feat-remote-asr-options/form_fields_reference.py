"""docling's own multipart form fields for ApiAsrOptions; remote-options-test expects these."""
from importlib.metadata import version
from docling.datamodel.pipeline_options_asr_model import ApiAsrOptions
from docling.pipeline import asr_transcriber as t

cls = next(c for c in vars(t).values() if isinstance(c, type) and hasattr(c, "_form_fields"))
print("docling-slim", version("docling-slim"), "class", cls.__name__)
cases = {
    "defaults": ApiAsrOptions(),
    "full": ApiAsrOptions(model="whisper-1", language="en", initial_prompt="NATO alphabet",
                          word_timestamps=True,
                          params={"a_flag": True, "b_off": False, "beam": 5, "response_format": "text",
                                  "temperature": 0.0, "top": 0.25, "vad": "silero"}),
    "empty strings omitted": ApiAsrOptions(model="", language="", initial_prompt=""),
}
print("default timeout", ApiAsrOptions().timeout)
for name, opts in cases.items():
    obj = cls.__new__(cls)
    obj.asr_options = opts
    print("==", name)
    for k, v in obj._form_fields():
        print(f"{k}={v}")
