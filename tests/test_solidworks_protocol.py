import base64
import json
import re
import unittest
from pathlib import Path

from ragtime.solidworks.protocol import (
    MAX_ENCODED_REQUEST_LENGTH,
    MAX_ENCODED_SCRIPT_REQUEST_LENGTH,
    SolidworksProtocolError,
    build_request,
    encode_request,
    error_response,
    parse_response,
)

HOST_REQUEST_ARGUMENT = Path(__file__).resolve().parents[1] / "dotnet/src/Ragtime.SolidWorks.Core/Wire/CallRequestArgument.cs"


class SolidworksProtocolTests(unittest.TestCase):
    def test_request_round_trips_as_compact_base64(self):
        request = build_request("status", None, {"deadline_seconds": 10})
        self.assertEqual(request["protocol"], 1)
        self.assertEqual(request["arguments"], {})
        self.assertEqual(json.loads(base64.b64decode(encode_request(request))), request)

    def test_request_rejects_oversized_payload(self):
        with self.assertRaises(SolidworksProtocolError):
            encode_request(build_request("status", {"data": "x" * 8000}, {}))

    def test_script_run_request_uses_the_larger_limit(self):
        arguments = {"source": "x" * 9000, "purpose": "build"}
        with self.assertRaises(SolidworksProtocolError):
            encode_request(build_request("status", arguments, {}))
        request = build_request("script.run", arguments, {})
        self.assertEqual(json.loads(base64.b64decode(encode_request(request))), request)

    def test_host_accepts_every_request_ragtime_sends(self):
        match = re.search(r"public const int MaxLength = (\d+);", HOST_REQUEST_ARGUMENT.read_text(encoding="utf-8"))
        if match is None:
            self.fail("CallRequestArgument.MaxLength was not found.")
        self.assertGreaterEqual(int(match.group(1)), max(MAX_ENCODED_REQUEST_LENGTH, MAX_ENCODED_SCRIPT_REQUEST_LENGTH))

    def test_response_uses_last_envelope_among_stdout_lines(self):
        first = error_response("status", "internal_error", "First")
        last = error_response("status", "host_not_running", "Last")
        stdout = f"banner\n{json.dumps(first)}\n{json.dumps(last)}\nnoise\n"
        self.assertEqual(parse_response(stdout), last)

    def test_non_envelope_output_is_rejected(self):
        with self.assertRaises(SolidworksProtocolError):
            parse_response('banner\n{"ok":true}\n')


if __name__ == "__main__":
    unittest.main()
