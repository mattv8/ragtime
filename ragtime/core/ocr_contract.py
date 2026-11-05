"""Versioned, non-secret contract shared by OCR ingestion paths."""

import ast
from hashlib import sha256
from pathlib import Path

OCR_PIPELINE_VERSION = 1

_OCR_CONTRACT_CONSTANTS = {
    "DEFAULT_MAX_IMAGE_DIMENSION",
    "DEFAULT_JPEG_QUALITY",
    "ANTHROPIC_MAX_ENCODED_IMAGE_BYTES",
    "OPENAI_COMPATIBLE_MAX_ENCODED_IMAGE_BYTES",
    "OCR_OUTPUT_BUDGET_CEILING",
    "OCR_FALLBACK_OUTPUT_BUDGET",
    "OCR_OUTPUT_BUDGET_CACHE_TTL_SECONDS",
    "OLLAMA_OCR_CONTEXT_LENGTH",
    "OLLAMA_OCR_THINK",
    "DEFAULT_VISION_OCR_TIMEOUT",
    "CLASSIFICATION_PROMPT",
    "CLASSIFICATION_SCHEMA",
    "OCR_WITH_CLASSIFICATION_PROMPT",
    "OCR_WITH_CLASSIFICATION_SCHEMA",
    "OCR_ONLY_PROMPT",
    "OCR_ONLY_SCHEMA",
}
_OCR_CONTRACT_FUNCTIONS = {
    "_load_image",
    "_process_image_content",
    "preprocess_image",
    "_source_mime_type",
    "_image_mime_type",
    "_prepare_vision_image_content",
    "_extract_ollama_response_text",
    "_request_vision_response_text",
    "_request_vision_response_once",
    "_retry_vision_request",
    "_reject_ollama_incomplete_response",
    "_resolve_ocr_output_budget",
    "_lookup_ocr_output_budget",
    "_bounded_vision_image",
    "_encoded_image_limit",
    "_parse_vision_response_json",
    "_extract_vision_text_lines",
}


def _semantic_source(source: bytes, *, include_recovery_module: bool) -> bytes:
    """Return semantic AST data, falling back to raw bytes for invalid source."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return source
    if include_recovery_module:
        return ast.dump(module, annotate_fields=True, include_attributes=False).encode()
    contract_nodes = []
    for node in module.body:
        is_ocr_helper = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_ollama_ocr_")
        is_contract_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _OCR_CONTRACT_FUNCTIONS
        is_contract_constant = isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in _OCR_CONTRACT_CONSTANTS for target in node.targets
        )
        if is_ocr_helper or is_contract_function or is_contract_constant:
            contract_nodes.append(ast.dump(node, annotate_fields=True, include_attributes=False))
    return "\n".join(contract_nodes).encode() if contract_nodes else source


def _ocr_pipeline_fingerprint_from_vision_source(source: bytes, recovery_source: bytes = b"") -> str:
    """Build an OCR semantic-contract fingerprint without importing its runtime."""
    vision_semantics = _semantic_source(source, include_recovery_module=False)
    recovery_semantics = _semantic_source(recovery_source, include_recovery_module=True)
    fingerprint_source = b"vision\0" + vision_semantics + b"\0recovery\0" + recovery_semantics
    digest = sha256(fingerprint_source).hexdigest()
    return f"{OCR_PIPELINE_VERSION}:{digest}"


def ocr_pipeline_fingerprint() -> str:
    """Return the version and extraction-implementation fingerprint for OCR jobs."""
    vision_source = Path(__file__).with_name("vision_models.py").read_bytes()
    recovery_source = Path(__file__).with_name("ocr_table_recovery.py").read_bytes()
    return _ocr_pipeline_fingerprint_from_vision_source(vision_source, recovery_source)
