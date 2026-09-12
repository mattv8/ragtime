import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker/scripts/extract_build_sbom.py"
DOCUMENT = {"spdxVersion": "SPDX-2.3", "SPDXID": "SPDXRef-DOCUMENT", "name": "ragtime"}


class ExtractBuildSbomTests(unittest.TestCase):
    def _extract(self, payload: object) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "inspect.json", Path(directory) / "sbom.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(["python3", str(SCRIPT), str(source), str(target)], text=True, capture_output=True)
            result.output = target.read_text(encoding="utf-8") if target.exists() else ""  # type: ignore[attr-defined]
            return result

    def test_extracts_direct_spdx_document(self) -> None:
        # This is the exact shape emitted by `imagetools inspect --format '{{json .SBOM}}'`.
        result = self._extract({"SPDX": DOCUMENT})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.output), DOCUMENT)  # type: ignore[attr-defined]

    def test_extracts_linux_amd64_platform_document(self) -> None:
        result = self._extract({"linux/amd64": {"SPDX": DOCUMENT}})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fails_closed_for_missing_or_invalid_documents(self) -> None:
        payloads: tuple[object, ...] = (
            {},
            {"linux/arm64": {"SPDX": DOCUMENT}},
            {"SPDX": {}},
            {"SPDX": {"spdxVersion": "2.3", "SPDXID": "SPDXRef-DOCUMENT"}},
            {"SPDX": {"spdxVersion": "SPDX-2.3", "SPDXID": "SPDXRef-not-document"}},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertNotEqual(self._extract(payload).returncode, 0)
