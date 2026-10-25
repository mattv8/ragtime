"""Enrollment lifecycle and network safety checks for remote clients."""

import asyncio
import ipaddress
import secrets
import socket
from hashlib import sha256
from typing import Any, cast

import psutil
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from ragtime.core.encryption import decrypt_secret
from ragtime.core.ssh import SSHConfig, test_ssh_connection
from ragtime.indexer.models import ToolType
from ragtime.indexer.repository import repository as tool_repository
from ragtime.remote_client.repository import remote_pairing_repository

MAX_PROBE_TARGETS = 8
PROBE_TIMEOUT_SECONDS = 5


def _most_actionable_probe_error(errors: list[str]) -> str:
    """Prefer errors that must not trigger an automatic tunnel fallback."""
    for error in errors:
        if _is_terminal_direct_probe_error(error):
            return error
    return errors[-1] if errors else "SSH connection failed"


def _is_terminal_direct_probe_error(error: str) -> bool:
    return any(word in error.lower() for word in ("authentication", "policy"))


def _new_keypair() -> tuple[str, str]:
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        cast(Encoding, Encoding.PEM),
        cast(PrivateFormat, PrivateFormat.OpenSSH),
        NoEncryption(),
    ).decode("ascii")
    public_key = private_key.public_key().public_bytes(cast(Encoding, Encoding.OpenSSH), cast(PublicFormat, PublicFormat.OpenSSH)).decode("ascii")
    return private_pem, public_key


def _safe_probe_address(address: str) -> str | None:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return None
    if parsed.is_loopback or parsed.is_link_local or parsed.is_multicast or parsed.is_unspecified:
        return None
    return str(parsed)


def _local_addresses() -> set[str]:
    addresses = {"127.0.0.1", "::1"}
    for interface_addresses in psutil.net_if_addrs().values():
        for interface_address in interface_addresses:
            if interface_address.family not in (socket.AF_INET, socket.AF_INET6):
                continue
            address = interface_address.address.split("%", 1)[0]
            try:
                addresses.add(str(ipaddress.ip_address(address)))
            except ValueError:
                continue

    try:
        hostname_addresses = socket.getaddrinfo(socket.gethostname(), None)
    except OSError:
        hostname_addresses = []
    for result in hostname_addresses:
        raw_address = result[4][0]
        if not isinstance(raw_address, str):
            continue
        address = raw_address.split("%", 1)[0]
        try:
            addresses.add(str(ipaddress.ip_address(address)))
        except ValueError:
            continue
    return addresses


class RemotePairingService:
    async def create(self, tool_type: ToolType, tool_id: str | None = None) -> tuple[dict[str, Any], str, str]:
        token = secrets.token_urlsafe(32)
        tunnel_credential = secrets.token_urlsafe(32)
        private_key, public_key = _new_keypair()
        pairing = await remote_pairing_repository.create(
            tool_type=tool_type,
            tool_id=tool_id,
            token=token,
            tunnel_credential=tunnel_credential,
            private_key=private_key,
            public_key=public_key,
        )
        return pairing, token, tunnel_credential

    async def regenerate(self, pairing_id: str) -> tuple[dict[str, Any], str, str] | None:
        token = secrets.token_urlsafe(32)
        tunnel_credential = secrets.token_urlsafe(32)
        private_key, public_key = _new_keypair()
        pairing = await remote_pairing_repository.regenerate(pairing_id, token, tunnel_credential, private_key, public_key)
        return (pairing, token, tunnel_credential) if pairing else None

    async def complete(self, token: str, nonce: str, facts: dict[str, Any], mode: str) -> dict[str, Any] | None:
        candidate = await remote_pairing_repository.get_by_token(token)
        if candidate is None:
            return None
        tool = await tool_repository.get_tool_config(str(candidate["toolId"]))
        if tool is None or (facts["os"] == "windows") != (tool.tool_type == ToolType.WINDOWS_SHELL):
            return None
        transport = "tunnel" if mode == "tunnel" else "direct"
        pairing = await remote_pairing_repository.redeem(token, nonce, facts, transport)
        if pairing is None:
            return None
        if pairing["status"] == "paired":
            return pairing
        facts = pairing.get("facts") or facts
        if transport == "direct" or mode == "auto":
            address, error = await self._verify_direct_ssh(pairing, facts)
            if address:
                transport = "direct"
                config = self._connection_config(pairing, facts, transport, address)
                return await self._mark_verified_pairing(str(pairing["id"]), config)
            if error and _is_terminal_direct_probe_error(error):
                return await remote_pairing_repository.fail(str(pairing["id"]), error)
            if mode == "direct":
                return await remote_pairing_repository.fail(str(pairing["id"]), error or "Direct SSH connection failed")
            pairing = await remote_pairing_repository.select_tunnel(str(pairing["id"])) or pairing
        # A tunnel is not paired until an authenticated relay session makes SSH
        # verification possible. Keep the draft disabled and pollable meanwhile.
        return pairing

    async def verify_tunnel(self, pairing_id: str) -> dict[str, Any] | None:
        """Finalize a consumed tunnel only after its relay endpoint passes SSH auth."""
        pairing = await remote_pairing_repository.get(pairing_id)
        if not pairing or str(getattr(pairing["status"], "value", pairing["status"])) != "consumed":
            return pairing
        facts = pairing.get("facts") or {}
        if not isinstance(facts, dict):
            return None
        config = self._connection_config(pairing, facts, "tunnel", None)
        config["key_content"] = decrypt_secret(str(pairing["encryptedPrivateKey"]))
        result = await asyncio.to_thread(test_ssh_connection, SSHConfig(**config))
        if not result.success:
            return await remote_pairing_repository.fail(pairing_id, result.stderr or "Tunnel SSH verification failed")
        return await self._mark_verified_pairing(pairing_id, config)

    async def complete_offline(self, pairing_id: str, nonce: str, facts: dict[str, Any]) -> dict[str, Any] | None:
        candidate = await remote_pairing_repository.get(pairing_id)
        if candidate is None:
            return None
        tool = await tool_repository.get_tool_config(str(candidate["toolId"]))
        if tool is None or (facts["os"] == "windows") != (tool.tool_type == ToolType.WINDOWS_SHELL):
            return None
        pairing = await remote_pairing_repository.redeem_offline(pairing_id, nonce, facts)
        if pairing is None:
            return None
        address, error = await self._verify_direct_ssh(pairing, facts)
        if not address:
            return await remote_pairing_repository.fail(pairing_id, error or "Offline direct SSH connection failed")
        return await self._mark_verified_pairing(pairing_id, self._connection_config(pairing, facts, "direct", address))

    async def _mark_verified_pairing(self, pairing_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
        pairing = await remote_pairing_repository.mark_paired(pairing_id, config)
        if pairing:
            from ragtime.remote_client.relay import relay_manager

            for old_pairing_id in pairing.pop("revokedPairingIds", []):
                await relay_manager.revoke(old_pairing_id)
        return pairing

    async def _verify_direct_ssh(self, pairing: dict[str, Any], facts: dict[str, Any]) -> tuple[str | None, str | None]:
        """Require pinned-host-key SSH authentication, not just a TCP listener."""
        addresses = facts.get("addresses", [])
        port = int(facts["sshd_port"])
        local_addresses = _local_addresses()
        candidates: list[str] = []
        for raw_address in addresses[:MAX_PROBE_TARGETS]:
            address = _safe_probe_address(raw_address)
            if address and address not in local_addresses:
                candidates.append(address)
        if not candidates:
            return None, "No routable SSH address was provided"

        async def probe(address: str) -> tuple[str | None, str | None]:
            try:
                config = SSHConfig(
                    host=address,
                    port=port,
                    user=str(facts["username"]),
                    key_content=decrypt_secret(str(pairing["encryptedPrivateKey"])),
                    host_keys=list(facts["host_keys"]),
                    pairing_id=str(pairing["id"]),
                    transport="direct",
                    shell="powershell" if facts["os"] == "windows" else "posix",
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
                result = await asyncio.wait_for(asyncio.to_thread(test_ssh_connection, config), timeout=PROBE_TIMEOUT_SECONDS + 1)
                return (address, None) if result.success else (None, result.stderr or "SSH connection failed")
            except (OSError, ValueError, asyncio.TimeoutError):
                return None, "SSH connection timed out or was unreachable"

        tasks = [asyncio.create_task(probe(address)) for address in candidates]
        try:
            errors: list[str] = []
            for task in asyncio.as_completed(tasks):
                address, error = await task
                if address:
                    return address, None
                if error:
                    errors.append(error)
            return None, _most_actionable_probe_error(errors)
        finally:
            for task in tasks:
                task.cancel()

    @staticmethod
    def _connection_config(pairing: dict[str, Any], facts: dict[str, Any], transport: str, address: str | None) -> dict[str, Any]:
        return {
            "host": address or "remote-relay",
            "port": int(facts["sshd_port"]),
            "user": facts["username"],
            "host_keys": list(facts["host_keys"]),
            "pairing_id": str(pairing["id"]),
            "transport": transport,
            "shell": "powershell" if facts["os"] == "windows" else "posix",
        }

    @staticmethod
    def artifact_hash(content: bytes) -> str:
        return sha256(content).hexdigest()


remote_pairing_service = RemotePairingService()
