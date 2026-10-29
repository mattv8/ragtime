import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import patch

from ragtime.remote_client.service import RemotePairingService, _local_addresses


def _interface_addresses():
    return {
        "lo0": [
            SimpleNamespace(family=socket.AF_INET, address="127.0.0.1"),
            SimpleNamespace(family=socket.AF_INET6, address="::1"),
        ],
        "en0": [
            SimpleNamespace(family=socket.AF_INET, address="192.0.2.10"),
            SimpleNamespace(family=socket.AF_INET6, address="2001:db8::10%en0"),
        ],
    }


def test_local_addresses_survive_hostname_resolution_failure_and_normalize_ipv6_scope():
    with (
        patch("psutil.net_if_addrs", return_value=_interface_addresses()),
        patch("ragtime.remote_client.service.socket.gethostname", return_value="unresolvable-host"),
        patch("ragtime.remote_client.service.socket.getaddrinfo", side_effect=socket.gaierror(8, "not known")),
    ):
        addresses = _local_addresses()

    assert addresses == {"127.0.0.1", "::1", "192.0.2.10", "2001:db8::10"}


def test_direct_probe_rejects_interface_addresses_when_hostname_resolution_fails():
    remote_address = "198.51.100.20"

    def successful_remote_probe(config):
        if config.host != remote_address:
            raise AssertionError(f"local interface address was probed: {config.host}")
        return SimpleNamespace(success=True, stderr="")

    facts = {
        "addresses": ["192.0.2.10", "2001:db8::10", remote_address],
        "sshd_port": 22,
        "username": "remote",
        "host_keys": ["ssh-ed25519 key"],
        "os": "linux",
    }
    pairing = {"id": "pairing-id", "encryptedPrivateKey": "encrypted-key"}

    with (
        patch("psutil.net_if_addrs", return_value=_interface_addresses()),
        patch("ragtime.remote_client.service.socket.gethostname", return_value="unresolvable-host"),
        patch("ragtime.remote_client.service.socket.getaddrinfo", side_effect=socket.gaierror(8, "not known")),
        patch("ragtime.remote_client.service.decrypt_secret", return_value="private-key"),
        patch("ragtime.remote_client.service.test_ssh_connection", side_effect=successful_remote_probe),
    ):
        result = asyncio.run(RemotePairingService()._verify_direct_ssh(pairing, facts))

    assert result == (remote_address, None)


def test_direct_probe_ignores_host_key_mismatch_when_another_address_matches():
    facts = {"addresses": ["198.51.100.10", "198.51.100.20"], "sshd_port": 22, "username": "remote", "host_keys": ["ssh-ed25519 key"], "os": "linux"}
    pairing = {"id": "pairing-id", "encryptedPrivateKey": "encrypted-key"}

    def probe(config):
        if config.host == "198.51.100.10":
            return SimpleNamespace(success=False, stderr="Host key verification failed")
        return SimpleNamespace(success=True, stderr="")

    with (
        patch("ragtime.remote_client.service._local_addresses", return_value=set()),
        patch("ragtime.remote_client.service.decrypt_secret", return_value="private-key"),
        patch("ragtime.remote_client.service.test_ssh_connection", side_effect=probe),
    ):
        assert asyncio.run(RemotePairingService()._verify_direct_ssh(pairing, facts)) == ("198.51.100.20", None)


def test_direct_probe_reports_host_key_mismatch_as_nonterminal_address_failure():
    facts = {"addresses": ["198.51.100.10"], "sshd_port": 22, "username": "remote", "host_keys": ["ssh-ed25519 key"], "os": "linux"}
    pairing = {"id": "pairing-id", "encryptedPrivateKey": "encrypted-key"}
    with (
        patch("ragtime.remote_client.service._local_addresses", return_value=set()),
        patch("ragtime.remote_client.service.decrypt_secret", return_value="private-key"),
        patch("ragtime.remote_client.service.test_ssh_connection", return_value=SimpleNamespace(success=False, stderr="Host key verification failed")),
    ):
        address, error = asyncio.run(RemotePairingService()._verify_direct_ssh(pairing, facts))
    assert address is None and error == "Host key verification failed"


def test_direct_probe_prioritizes_pinned_target_authentication_failure():
    facts = {"addresses": ["198.51.100.10", "198.51.100.20"], "sshd_port": 22, "username": "remote", "host_keys": ["ssh-ed25519 key"], "os": "linux"}
    pairing = {"id": "pairing-id", "encryptedPrivateKey": "encrypted-key"}

    def probe(config):
        message = "Host key verification failed" if config.host.endswith("10") else "Authentication failed"
        return SimpleNamespace(success=False, stderr=message)

    with (
        patch("ragtime.remote_client.service._local_addresses", return_value=set()),
        patch("ragtime.remote_client.service.decrypt_secret", return_value="private-key"),
        patch("ragtime.remote_client.service.test_ssh_connection", side_effect=probe),
    ):
        assert asyncio.run(RemotePairingService()._verify_direct_ssh(pairing, facts)) == (None, "Authentication failed")
