import sys
import types
from typing import cast
from unittest.mock import patch

import paramiko

from ragtime.core.ssh import SSHConfig, _create_ssh_client, execute_ssh_command


def test_pinned_host_key_mismatch_rejects_before_authentication():
    key = paramiko.RSAKey.generate(1024)
    enrolled = f"{key.get_name()} {key.get_base64()}"

    class Client:
        def __init__(self):
            self.policy: paramiko.MissingHostKeyPolicy | None = None
            self._host_keys = paramiko.HostKeys()

        def get_host_keys(self):
            return self._host_keys

        def set_missing_host_key_policy(self, policy):
            self.policy = policy

        def _log(self, *_args):
            pass

        def connect(self, **kwargs):
            assert "password" in kwargs
            assert self.policy is not None
            self.policy.missing_host_key(cast(paramiko.SSHClient, self), kwargs["hostname"], paramiko.RSAKey.generate(1024))

        def close(self):
            pass

    with patch("ragtime.core.ssh.paramiko.SSHClient", Client):
        try:
            _create_ssh_client(SSHConfig(host="host", user="user", password="secret", host_keys=[enrolled]))
        except paramiko.SSHException:
            pass
        else:
            raise AssertionError("host key mismatch was accepted")


def test_offline_relay_does_not_fall_back_to_stale_endpoint():
    relay_module = types.ModuleType("ragtime.remote_client.relay")
    setattr(relay_module, "relay_manager", types.SimpleNamespace(endpoint_for=lambda _pairing_id: None))
    package = types.ModuleType("ragtime.remote_client")
    with patch.dict(sys.modules, {"ragtime.remote_client": package, "ragtime.remote_client.relay": relay_module}):
        result = execute_ssh_command(
            SSHConfig(
                host="stale-host",
                user="user",
                password="secret",
                pairing_id="pairing",
                host_keys=["ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAgQDc"],
                transport="tunnel",
            ),
            "echo ok",
        )
    assert not result.success
    assert "offline" in result.stderr.lower()
