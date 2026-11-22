import base64
import json
import unittest

from ragtime.solidworks.protocol import (
    SolidworksProtocolError,
    build_request,
    encode_request,
    error_response,
    parse_response,
)


class SolidworksProtocolTests(unittest.TestCase):
    def test_request_round_trips_as_compact_base64(self):
        request = build_request("status", None, {"deadline_seconds": 10})
        self.assertEqual(request["protocol"], 1)
        self.assertEqual(request["arguments"], {})
        self.assertEqual(json.loads(base64.b64decode(encode_request(request))), request)

    def test_request_rejects_oversized_payload(self):
        with self.assertRaises(SolidworksProtocolError):
            encode_request(build_request("status", {"data": "x" * 8000}, {}))

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
