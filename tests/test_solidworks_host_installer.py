import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ragtime.solidworks import host_installer as installer
from ragtime.solidworks.host_artifact import HostArtifact
from ragtime.solidworks.host_status import SolidworksHostStatusService, empty_host_status

LOCAL_APP_DATA = r"C:\Users\name'$(evil)\AppData\Local"
STAGED = "RAGTIME_STAGING:ok"


def result(stdout="", exit_code=0, stderr=""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


def envelope():
    return json.dumps({"protocol": 1, "id": "1", "operation": "status", "ok": True, "result": {}})


def probe_line(present: bool) -> str:
    data = {
        "exe_present": present,
        "host_version": "1.0" if present else None,
        "account": {"name": "DOMAIN\\user", "is_administrator": True},
        "interactive": True,
        "installed_versions": [2024],
    }
    return json.dumps(data)


class Transport:
    def __init__(self, responses, upload_error=False):
        self.responses = iter(responses)
        self.upload_error = upload_error
        self.calls = []
        self.timeouts = []

    async def run_powershell(self, script, *, timeout_seconds):
        self.calls.append(("script", script))
        self.timeouts.append(timeout_seconds)
        return next(self.responses)

    async def upload_file(self, local_path, remote_path, *, timeout_seconds):
        self.calls.append(("upload", local_path, remote_path))
        self.timeouts.append(timeout_seconds)
        if self.upload_error:
            raise OSError("secret")

    async def call(self, request, *, timeout_seconds):
        self.calls.append(("call", request))
        return {"ok": False}


class Status:
    async def host_status(self, config, *, probe_only=False):
        return {**empty_host_status(), "reachable": True}


class HostInstallerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "host.zip"
        path.write_bytes(b"zip")
        self.artifact = HostArtifact("1", 1, "host.zip", hashlib.sha256(b"zip").hexdigest(), 3, path)
        patches = [
            mock.patch.object(installer, "get_host_artifact", return_value=self.artifact),
            mock.patch("ragtime.solidworks.host_status.get_host_artifact", return_value=None),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def service(self, transport, status=None):
        return installer.SolidworksHostInstaller(lambda config: transport, status or Status())

    def remote_path(self):
        return "C:/Users/name'$(evil)/AppData/Local/RagtimeSolidWorks/staging/host-" + self.artifact.sha256[:16] + ".zip"

    async def test_install_order_and_untrusted_path(self):
        transport = Transport([result(json.dumps(LOCAL_APP_DATA)), result(STAGED), result(envelope())])
        status = await self.service(transport).install_host({})
        self.assertIsNone(status["error"])
        self.assertEqual([call[0] for call in transport.calls], ["script", "script", "upload", "script"])
        self.assertEqual(transport.calls[2][1:], (self.artifact.path, self.remote_path()))
        self.assertTrue(all(LOCAL_APP_DATA not in call[1] for call in transport.calls if call[0] == "script"))
        script = transport.calls[-1][1]
        self.assertEqual(script, installer.INSTALL_SCRIPT.replace("__SHA__", self.artifact.sha256).replace("__PREFIX__", self.artifact.sha256[:16]))
        self.assertNotIn("__SHA__", script)
        self.assertNotIn("__PREFIX__", script)
        # Length guard: the payload travels over stdin (ragtime/core/ssh.py _encode_powershell_payload), not on the command line.
        self.assertLess(len(script), 6000)
        self.assertLess(len(installer.UNINSTALL_SCRIPT), 4000)
        self.assertLess(len(installer.STAGING_SCRIPT), 4000)

    async def test_invalid_path_rejected(self):
        paths = [r"\\server\share", r"\\?\C:\local", r"C:relative", "C:\\bad\npath", r"C:\foo\..\bar"]
        for path in paths:
            with self.subTest(path=path):
                transport = Transport([result(json.dumps(path))])
                status = await self.service(transport).install_host({})
                self.assertEqual(status["error"]["code"], "install_failed")
                self.assertEqual(len(transport.calls), 1)

    async def test_install_sentinels(self):
        cases = [
            ("hash_mismatch", "hash_mismatch"),
            ("locked", "install_failed"),
            ("install_failed", "install_failed"),
        ]
        for sentinel, code in cases:
            with self.subTest(sentinel=sentinel):
                transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("RAGTIME_INSTALL:" + sentinel, 1)])
                status = await self.service(transport).install_host({})
                self.assertEqual(status["error"]["code"], code)
        locked = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("RAGTIME_INSTALL:locked", 1)])
        self.assertIn("locked", (await self.service(locked).install_host({}))["error"]["message"])

    async def test_recovery_locked_sentinel_maps_to_install_failed_with_locked_message(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("RAGTIME_INSTALL:locked", 1)])
        status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "install_failed")
        self.assertEqual(status["error"]["message"], installer.MESSAGES["locked"])

    async def test_rolled_back_failure_reports_previous_version_restored(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("RAGTIME_INSTALL:rolled_back", 1)])
        status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "install_failed")
        self.assertIn("previous version was restored", status["error"]["message"])

    async def test_install_success_with_nonzero_exit_and_valid_status_line(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("noise\n" + envelope(), 1)])
        status = await self.service(transport).install_host({})
        self.assertIsNone(status["error"])

    async def test_install_sentinel_wins_over_zero_exit_code(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result(envelope() + "\nRAGTIME_INSTALL:install_failed", 0)])
        status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "install_failed")

    async def test_install_without_status_line_fails(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("garbage", 0)])
        status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "install_failed")

    async def test_install_without_output_uses_ssh_classification(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("", -1, stderr="ssh connection refused")])
        status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "ssh_unreachable")

    async def test_update_expand_failure_never_uninstalls_before_expanding(self):
        script = installer.INSTALL_SCRIPT
        self.assertLess(script.index("Get-FileHash"), script.index("Expand-Archive"))
        self.assertLess(script.index("Expand-Archive"), script.index("& $exe uninstall"))
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result("RAGTIME_INSTALL:install_failed", 1)])
        status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "install_failed")
        self.assertEqual(status["error"]["message"], installer.MESSAGES["install_failed"])

    async def test_staging_requires_its_sentinel(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED, 1), result(envelope())])
        status = await self.service(transport).install_host({})
        self.assertIsNone(status["error"])
        self.assertEqual(transport.calls[1][0], "script")
        for staging in [result("", 0), result("RAGTIME_STAGING:failed", 1)]:
            with self.subTest(staging=staging.stdout):
                transport = Transport([result(json.dumps(r"C:\local")), staging])
                status = await self.service(transport).install_host({})
                self.assertEqual(status["error"]["code"], "install_failed")
                self.assertFalse(any(call[0] == "upload" for call in transport.calls))

    async def test_upload_failure(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED)], upload_error=True)
        status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "upload_failed")
        self.assertNotIn("secret", str(status))

    async def test_artifact_mismatch_refuses_install_before_any_ssh_call(self):
        transport = Transport([])
        with mock.patch.object(installer, "verify_host_artifact", return_value=False):
            status = await self.service(transport).install_host({})
        self.assertEqual(status["error"]["code"], "install_failed")
        self.assertIn("manifest", status["error"]["message"])
        self.assertEqual(transport.calls, [])

    async def test_timeouts_are_named_positive_constants(self):
        transport = Transport([result(json.dumps(r"C:\local")), result(STAGED), result(envelope())])
        await self.service(transport).install_host({})
        self.assertEqual(transport.timeouts, [30, 30, 60, 90])
        self.assertEqual(
            transport.timeouts,
            [
                installer.RESOLVE_TIMEOUT_SECONDS,
                installer.STAGING_TIMEOUT_SECONDS,
                installer.UPLOAD_TIMEOUT_SECONDS,
                installer.INSTALL_TIMEOUT_SECONDS,
            ],
        )
        self.assertTrue(all(seconds > 0 for seconds in transport.timeouts))

    async def test_uninstall_failure_is_an_error_without_warning(self):
        transport = Transport([result("RAGTIME_UNINSTALL:uninstall_failed", 1)])
        status = await self.service(transport).uninstall_host({})
        self.assertEqual(status["error"]["code"], "uninstall_failed")
        self.assertIsNone(status["warning"])

    async def test_uninstall_without_output_uses_ssh_classification(self):
        transport = Transport([result("", -1, stderr="ssh connection refused")])
        status = await self.service(transport).uninstall_host({})
        self.assertEqual(status["error"]["code"], "ssh_unreachable")

    async def test_uninstall_locked_is_a_warning_and_never_calls_the_host(self):
        transport = Transport([result("RAGTIME_UNINSTALL:locked"), result(probe_line(present=True))])
        installer_service = installer.SolidworksHostInstaller(lambda config: transport, SolidworksHostStatusService(lambda config: transport))
        status = await installer_service.uninstall_host({})
        self.assertIsNone(status["error"])
        self.assertIn("could not be deleted", status["warning"])
        self.assertTrue(status["installed"])
        self.assertFalse(status["running"])
        self.assertEqual([call[0] for call in transport.calls], ["script", "script"])

    async def test_clean_uninstall_reports_not_installed(self):
        transport = Transport([result("RAGTIME_UNINSTALL:ok"), result(probe_line(present=False))])
        installer_service = installer.SolidworksHostInstaller(lambda config: transport, SolidworksHostStatusService(lambda config: transport))
        status = await installer_service.uninstall_host({})
        self.assertFalse(status["installed"])
        self.assertIsNone(status["error"])
        self.assertIsNone(status["warning"])
        self.assertFalse(any(call[0] == "call" for call in transport.calls))

    async def test_every_status_carries_warning_key(self):
        outcomes = [
            await self.service(Transport([result(json.dumps(r"C:\local")), result(STAGED), result(envelope())])).install_host({}),
            await self.service(Transport([result(json.dumps(r"C:\local")), result(STAGED), result("", 1)])).install_host({}),
            await self.service(Transport([result("RAGTIME_UNINSTALL:ok")])).uninstall_host({}),
            await self.service(Transport([result("RAGTIME_UNINSTALL:uninstall_failed", 1)])).uninstall_host({}),
        ]
        for status in outcomes:
            with self.subTest(status=status["error"]):
                self.assertIn("warning", status)

    async def test_missing_artifact(self):
        with mock.patch.object(installer, "get_host_artifact", return_value=None):
            with self.assertRaises(installer.HostArtifactUnavailableError):
                await self.service(Transport([])).install_host({})


class InstallScriptTemplateTests(unittest.TestCase):
    def test_uninstall_script_ends_every_success_path_with_exit_zero(self):
        self.assertIn("RAGTIME_UNINSTALL:ok'\n  exit 0", installer.UNINSTALL_SCRIPT)
        self.assertIn("RAGTIME_UNINSTALL:locked'\n      exit 0", installer.UNINSTALL_SCRIPT)

    def test_recovery_of_interrupted_run_precedes_hash_check_and_any_deletion(self):
        script = installer.INSTALL_SCRIPT
        hash_check = script.index("Get-FileHash")
        recovery = script.index("-not (Test-Path -LiteralPath $exe -PathType Leaf) -and")
        self.assertLess(recovery, hash_check)
        self.assertLess(script.index("Rename-Item -LiteralPath $old -NewName 'host'"), hash_check)
        self.assertLess(script.index("RAGTIME_INSTALL:locked"), hash_check)

    def test_host_old_is_deleted_only_at_start_with_working_host_or_after_success(self):
        lines = installer.INSTALL_SCRIPT.splitlines()
        hash_line = next(index for index, line in enumerate(lines) if "Get-FileHash" in line)
        success_line = next(index for index, line in enumerate(lines) if "Write-Output $status" in line)
        deletions = [index for index, line in enumerate(lines) if re.search(r"Remove-Item -LiteralPath \$old\b", line)]
        self.assertEqual(len(deletions), 2)
        for index in deletions:
            self.assertTrue(index < hash_line or index > success_line, lines[index])

    def test_failed_host_is_renamed_aside_not_recursively_deleted(self):
        script = installer.INSTALL_SCRIPT
        failure = script[script.index("$output = & $exe install") :]
        self.assertIn("Rename-Item -LiteralPath $hostDir -NewName 'host.failed'", failure)
        self.assertNotIn("Remove-Item -LiteralPath $hostDir", failure)
        self.assertLess(failure.index("NewName 'host.failed'"), failure.index("NewName 'host'"))

    def assert_in_order(self, script: str, patterns: tuple[str, ...]) -> None:
        positions = []
        for pattern in patterns:
            match = re.search(pattern, script)
            assert match is not None, pattern
            positions.append(match.start())
        self.assertEqual(positions, sorted(positions))

    def assert_wait_and_retry_limits(self, script: str) -> None:
        for name in ("exitWaitSeconds", "pollMs", "retryTries", "retryDelayMs"):
            with self.subTest(limit=name):
                self.assertRegex(script, rf"(?m)^\${name} = \d+$")
                self.assertGreater(script.count(f"${name}"), 1)
        self.assertIn("catch { $false }", script)

    def test_install_waits_for_host_exit_then_retries_rename(self):
        script = installer.INSTALL_SCRIPT
        self.assert_in_order(script, (r"& \$exe uninstall", r"Get-Process\b", r"for \(\$i = 1; ; \$i\+\+\)", r"\$output = & \$exe install"))
        self.assert_wait_and_retry_limits(script)

    def test_uninstall_waits_for_host_exit_then_retries_removal(self):
        script = installer.UNINSTALL_SCRIPT
        self.assert_in_order(script, (r"& \$exe uninstall", r"Get-Process\b", r"for \(\$i = 1; ; \$i\+\+\)", r"Remove-Item -LiteralPath \$hostDir"))
        self.assert_wait_and_retry_limits(script)
        self.assertRegex(script, r"-Name 'ragtime-solidworks-host'")


if __name__ == "__main__":
    unittest.main()
