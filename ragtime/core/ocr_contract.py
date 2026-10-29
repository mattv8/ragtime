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
}


def _ocr_pipeline_fingerprint_from_vision_source(source: bytes) -> str:
    """Build an OCR semantic-contract fingerprint without importing its runtime."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return f"{OCR_PIPELINE_VERSION}:{sha256(source).hexdigest()}"
    contract_nodes = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _OCR_CONTRACT_FUNCTIONS:
            contract_nodes.append(ast.dump(node, annotate_fields=True, include_attributes=False))
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in _OCR_CONTRACT_CONSTANTS for target in node.targets):
            contract_nodes.append(ast.dump(node, annotate_fields=True, include_attributes=False))
    fingerprint_source = "\n".join(contract_nodes).encode() if contract_nodes else source
    digest = sha256(fingerprint_source).hexdigest()
    return f"{OCR_PIPELINE_VERSION}:{digest}"


def ocr_pipeline_fingerprint() -> str:
    """Return the version and extraction-implementation fingerprint for OCR jobs."""
    vision_source = Path(__file__).with_name("vision_models.py").read_bytes()
    return _ocr_pipeline_fingerprint_from_vision_source(vision_source)
