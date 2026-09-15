import httpx

from ragtime.core.provider_errors import classify_provider_error, provider_error_message


def test_classifies_http_402_without_exposing_response_text() -> None:
    request = httpx.Request("POST", "https://provider.example/chat")
    error = httpx.HTTPStatusError(
        "sensitive provider detail", request=request, response=httpx.Response(402, request=request)
    )

    assert classify_provider_error(error, "openrouter") == "payment_required"
    assert "sensitive" not in provider_error_message("payment_required")


def test_classifies_nested_and_sse_payment_error_payloads() -> None:
    nested = {"error": {"metadata": {"error_type": "payment_required"}}}
    sse = 'event: error\ndata: {"error":{"metadata":{"error_type":"insufficient_credits"}}}\n\n'

    assert classify_provider_error(nested) == "payment_required"
    assert classify_provider_error(sse) == "payment_required"


def test_does_not_treat_generic_quota_or_rate_limits_as_payment() -> None:
    assert classify_provider_error({"error": {"message": "quota exceeded"}}) is None
    assert classify_provider_error({"error": {"code": "rate_limit_exceeded"}}) is None
    assert provider_error_message("anything_else") == "The provider request could not be completed."
