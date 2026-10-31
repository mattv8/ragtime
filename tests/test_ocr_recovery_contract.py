"""Regression coverage for OCR table-recovery contract changes."""

import re

import httpx
import pytest

from ragtime.core import ocr_contract
from ragtime.core.ocr_contract import _ocr_pipeline_fingerprint_from_vision_source, ocr_pipeline_fingerprint
from ragtime.core.ocr_table_recovery import OllamaOcrIncompleteError
from ragtime.indexer.document_parser import _structured_vision_failure_detail


def test_fingerprint_tracks_recovery_module_semantics() -> None:
    vision_source = b'RECOVERY_PROMPT = "original"\n'
    recovery_source = b'RECOVERY_PENALTY = 1.05\ndef render_table(rows):\n    return "|".join(rows)\n'

    assert _ocr_pipeline_fingerprint_from_vision_source(vision_source, recovery_source) != _ocr_pipeline_fingerprint_from_vision_source(
        vision_source, recovery_source.replace(b"1.05", b"1.10")
    )
    assert _ocr_pipeline_fingerprint_from_vision_source(vision_source, recovery_source) != _ocr_pipeline_fingerprint_from_vision_source(
        vision_source, recovery_source.replace(b'"|".join', b'" || ".join')
    )


def test_fingerprint_ignores_recovery_comments_and_whitespace() -> None:
    vision_source = b"def _ollama_ocr_recover_table(payload):\n    return payload\n"
    recovery_source = b"RECOVERY_PENALTY = 1.05\ndef render_table(rows):\n    return '|'.join(rows)\n"
    reformatted_recovery_source = b"\n# Recovery contract\nRECOVERY_PENALTY = 1.05\n\n\ndef render_table(rows):\n\n    return '|'.join(rows)\n"

    assert _ocr_pipeline_fingerprint_from_vision_source(vision_source, recovery_source) == _ocr_pipeline_fingerprint_from_vision_source(
        vision_source, reformatted_recovery_source
    )


def test_fingerprint_tracks_ollama_recovery_helpers_but_excludes_selectors() -> None:
    source = b"""\ndef _ollama_ocr_recover_table(payload):
    return "retry"
def select_model():
    return "first"
"""

    assert _ocr_pipeline_fingerprint_from_vision_source(source) != _ocr_pipeline_fingerprint_from_vision_source(source.replace(b'"retry"', b'"fail"'))
    assert _ocr_pipeline_fingerprint_from_vision_source(source) == _ocr_pipeline_fingerprint_from_vision_source(source.replace(b'"first"', b'"second"'))


def test_fingerprint_ignores_ollama_helper_comments_and_whitespace() -> None:
    source = b"def _ollama_ocr_recover_table(payload):\n    return payload\n"
    reformatted_source = b"\n# This helper retries typed table extraction.\ndef _ollama_ocr_recover_table(payload):\n\n    return payload\n"

    assert _ocr_pipeline_fingerprint_from_vision_source(source) == _ocr_pipeline_fingerprint_from_vision_source(reformatted_source)


def test_fingerprint_tracks_existing_request_flow_and_limits() -> None:
    source = b"""OCR_FALLBACK_OUTPUT_BUDGET = 4096
DEFAULT_VISION_OCR_TIMEOUT = 180.0
async def _request_vision_response_once():
    return "first"
async def _retry_vision_request():
    return "retry"
"""
    fingerprint = _ocr_pipeline_fingerprint_from_vision_source(source)

    for changed_source in (
        source.replace(b'"first"', b'"second"'),
        source.replace(b'"retry"', b'"stop"'),
        source.replace(b"4096", b"8192"),
        source.replace(b"180.0", b"240.0"),
    ):
        assert fingerprint != _ocr_pipeline_fingerprint_from_vision_source(changed_source)


@pytest.mark.parametrize(
    "helper_name",
    ["_bounded_vision_image", "_encoded_image_limit", "_parse_vision_response_json", "_extract_vision_text_lines"],
)
def test_fingerprint_tracks_extraction_helper_dependencies(helper_name: str) -> None:
    anchor = 'async def _request_vision_response_once():\n    return "anchor"\n'
    helper = f'def {helper_name}(value):\n    return "first"\n'
    selector = 'def select_model():\n    return "first"\n'
    source = (anchor + helper + selector).encode()
    fingerprint = _ocr_pipeline_fingerprint_from_vision_source(source)

    assert fingerprint != _ocr_pipeline_fingerprint_from_vision_source((anchor + helper.replace('"first"', '"second"') + selector).encode())
    assert fingerprint == _ocr_pipeline_fingerprint_from_vision_source((anchor + helper + selector.replace('"first"', '"second"')).encode())


def test_fingerprint_tracks_ocr_prompt_and_schema() -> None:
    source = b"""OCR_ONLY_PROMPT = "extract all text"
OCR_ONLY_SCHEMA = {"type": "object", "required": ["extracted_text"]}
"""
    fingerprint = _ocr_pipeline_fingerprint_from_vision_source(source)

    assert fingerprint != _ocr_pipeline_fingerprint_from_vision_source(source.replace(b"all text", b"tables only"))
    assert fingerprint != _ocr_pipeline_fingerprint_from_vision_source(source.replace(b"extracted_text", b"tables"))


def test_fingerprint_invalid_source_includes_recovery_source() -> None:
    vision_source = b"def broken("
    recovery_source = b"def recovery_broken("

    assert _ocr_pipeline_fingerprint_from_vision_source(vision_source, recovery_source) != _ocr_pipeline_fingerprint_from_vision_source(
        vision_source, recovery_source + b"# changed"
    )


def test_real_fingerprint_is_stable() -> None:
    assert ocr_pipeline_fingerprint() == ocr_pipeline_fingerprint()


def test_real_fingerprint_has_versioned_sha256_format_and_reads_both_sources(monkeypatch) -> None:
    sources = {
        "vision_models.py": b'OCR_ONLY_PROMPT = "extract"\n',
        "ocr_table_recovery.py": b"RECOVERY_PENALTY = 1.05\n",
    }
    monkeypatch.setattr(ocr_contract.Path, "read_bytes", lambda path: sources[path.name])

    fingerprint = ocr_contract.ocr_pipeline_fingerprint()
    assert re.fullmatch(rf"{ocr_contract.OCR_PIPELINE_VERSION}:[0-9a-f]{{64}}", fingerprint)

    sources["vision_models.py"] = b'OCR_ONLY_PROMPT = "extract tables"\n'
    assert fingerprint != ocr_contract.ocr_pipeline_fingerprint()

    sources["vision_models.py"] = b'OCR_ONLY_PROMPT = "extract"\n'
    sources["ocr_table_recovery.py"] = b"RECOVERY_PENALTY = 1.10\n"
    assert fingerprint != ocr_contract.ocr_pipeline_fingerprint()


def test_incomplete_ollama_diagnostics_expose_only_safe_reason_and_counts() -> None:
    exhausted = OllamaOcrIncompleteError("length", eval_count=11, prompt_eval_count=7)
    unsafe = OllamaOcrIncompleteError("untrusted SECRET_RAW_OCR", eval_count=11, prompt_eval_count=7)

    exhausted_detail = _structured_vision_failure_detail(exhausted, "ollama")
    unsafe_detail = _structured_vision_failure_detail(unsafe, "ollama")

    assert "category=output_exhausted" in exhausted_detail
    assert "reason=length" in exhausted_detail
    assert "eval_count=11" in exhausted_detail
    assert "prompt_eval_count=7" in exhausted_detail
    assert "category=incomplete_response" in unsafe_detail
    assert "SECRET_RAW_OCR" not in unsafe_detail
    assert "untrusted" not in unsafe_detail


def test_incomplete_ollama_diagnostics_classify_known_reasons_and_drop_invalid_counts() -> None:
    for reason, counter in (("error", None), ("load", True), ("malformed", -1), ("incomplete", -1)):
        detail = _structured_vision_failure_detail(OllamaOcrIncompleteError(reason, eval_count=counter, prompt_eval_count=counter), "ollama")

        assert f"provider=ollama, category=incomplete_response, reason={reason}" == detail
        assert "eval_count" not in detail
        assert "prompt_eval_count" not in detail


def test_structured_diagnostics_preserve_http_timeout_and_generic_categories() -> None:
    request = httpx.Request("POST", "https://ocr.test/api/generate")
    http_error = httpx.HTTPStatusError("bad gateway", request=request, response=httpx.Response(502, request=request))

    assert _structured_vision_failure_detail(http_error, "ollama") == "provider=ollama, category=http_status, status=502"
    assert _structured_vision_failure_detail(httpx.ReadTimeout("SECRET_URL"), "ollama") == "provider=ollama, category=timeout"
    assert _structured_vision_failure_detail(ValueError("SECRET_RAW_OCR"), "ollama") == "provider=ollama, category=extraction_failed, error_type=ValueError"
