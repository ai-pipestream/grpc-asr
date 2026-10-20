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
    # Keys in sorted order, so docling's dict order equals the sorted order
    # remote-options-test sends.
    "floats": ApiAsrOptions(params={
        "f01": 1e5, "f02": 1e15, "f03": 1e16, "f04": 1e-4, "f05": 1e-5,
        "f06": 123456.789, "f07": -0.0, "f08": 1.5e-7, "f09": 0.1, "f10": 1e22,
        "f11": -2.5e-300, "f12": 9007199254740993.0, "f13": 5e-324, "f14": 1.7976931348623157e308,
    }),
}
print("default timeout", ApiAsrOptions().timeout)
for name, opts in cases.items():
    obj = cls.__new__(cls)
    obj.asr_options = opts
    print("==", name)
    for k, v in obj._form_fields():
        print(f"{k}={v}")
