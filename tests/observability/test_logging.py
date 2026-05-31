import json
import logging

from medical_qa_platform.observability.logging import JsonFormatter, get_logger


def test_formatter_emits_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.trace_id = "abc"
    parsed = json.loads(formatter.format(record))
    assert parsed["message"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["trace_id"] == "abc"


def test_get_logger_returns_logger():
    assert isinstance(get_logger("x"), logging.Logger)
