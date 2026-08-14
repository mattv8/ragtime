from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import mock

from jose import jwt  # type: ignore[import-untyped]

from ragtime.config.settings import settings
from ragtime.userspace.runtime_service import UserSpaceRuntimeService


def _decode(token: str) -> dict:
    return jwt.decode(token, settings.encryption_key, algorithms=[settings.jwt_algorithm])


class RuntimeBridgeEnvTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = UserSpaceRuntimeService()

    def test_bridge_env_contains_url_and_token(self) -> None:
        env = self.service._build_runtime_bridge_env("ws-1", "sess-1")
        self.assertIn("RAGTIME_BRIDGE_URL", env)
        self.assertTrue(env["RAGTIME_BRIDGE_URL"].startswith("http"))
        self.assertIn("/runtime-bridge", env["RAGTIME_BRIDGE_URL"])
        claims = _decode(env["RAGTIME_BRIDGE_TOKEN"])
        self.assertEqual(claims["workspace_id"], "ws-1")
        self.assertEqual(claims["session_id"], "sess-1")

    def test_bridge_env_exposes_only_bridge_url_and_token(self) -> None:
        env = self.service._build_runtime_bridge_env("ws-1", "sess-1")
        self.assertEqual(set(env.keys()), {"RAGTIME_BRIDGE_URL", "RAGTIME_BRIDGE_TOKEN"})
        dumped = str(env)
        self.assertNotIn("bearer_token", dumped)
        self.assertNotIn("api_key", dumped)
        self.assertNotIn("basic_password", dumped)

    def test_runtime_bridge_origin_prefers_explicit_override(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", " https://bridge.internal:9443/runtime-bridge "),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://runtime:8090"),
            ),
            mock.patch.object(
                UserSpaceRuntimeService,
                "_discover_runtime_bridge_host",
                return_value="172.18.0.3",
            ),
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "https://bridge.internal:9443")

    def test_runtime_bridge_origin_ignores_invalid_explicit_override_and_logs_warning(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", "ragtime:8000"),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch.object(settings, "enable_https", False),
            mock.patch.object(settings, "port", 8000),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://runtime:8090"),
            ),
            mock.patch.object(
                UserSpaceRuntimeService,
                "_discover_runtime_bridge_host",
                return_value="172.18.0.3",
            ),
            mock.patch("ragtime.userspace.runtime_service.logger.warning") as warning,
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "http://172.18.0.3:8000")
        warning.assert_called_once()
        warning_message = warning.call_args.args[0]
        self.assertIn("Ignoring invalid RUNTIME_BRIDGE_BASE_URL", warning_message)
        self.assertNotIn("ragtime:8000", warning_message)

    def test_runtime_bridge_origin_discovers_internal_origin_for_private_manager_peer(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", ""),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch.object(settings, "enable_https", False),
            mock.patch.object(settings, "port", 8000),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://runtime:8090"),
            ),
            mock.patch.object(
                UserSpaceRuntimeService,
                "_discover_runtime_bridge_host",
                return_value="172.18.0.3",
            ),
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "http://172.18.0.3:8000")

    def test_runtime_bridge_origin_falls_back_to_public_origin_for_public_manager_peer(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", ""),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch.object(settings, "enable_https", False),
            mock.patch.object(settings, "port", 8000),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://203.0.113.10:8090"),
            ),
            mock.patch.object(
                UserSpaceRuntimeService,
                "_discover_runtime_bridge_host",
                return_value=None,
            ),
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "https://public.example.com")

    def test_runtime_bridge_origin_skips_discovery_when_https_enabled_without_override(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", ""),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch.object(settings, "enable_https", True),
            mock.patch.object(settings, "port", 8443),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://runtime:8090"),
            ),
            mock.patch.object(
                UserSpaceRuntimeService,
                "_discover_runtime_bridge_host",
                return_value="fd00::1",
            ) as discover,
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "https://public.example.com")
        discover.assert_not_called()

    def test_runtime_bridge_origin_uses_https_default_manager_port_for_probe(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", ""),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch.object(settings, "enable_https", False),
            mock.patch.object(settings, "port", 8000),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="https://runtime"),
            ),
            mock.patch.object(
                UserSpaceRuntimeService,
                "_discover_runtime_bridge_host",
                return_value="172.18.0.3",
            ) as discover,
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "http://172.18.0.3:8000")
        discover.assert_called_once_with("runtime", 443)

    def test_runtime_bridge_origin_falls_back_to_public_origin_when_manager_is_loopback(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", ""),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://127.0.0.1:8090"),
            ),
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "https://public.example.com")

    def test_runtime_bridge_origin_falls_back_to_public_origin_when_discovery_fails(self) -> None:
        with (
            mock.patch.object(settings, "runtime_bridge_base_url", ""),
            mock.patch.object(settings, "external_base_url", "https://public.example.com/app"),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://runtime:8090"),
            ),
            mock.patch.object(
                UserSpaceRuntimeService,
                "_discover_runtime_bridge_host",
                return_value=None,
            ),
        ):
            origin = self.service._runtime_bridge_control_plane_origin()

        self.assertEqual(origin, "https://public.example.com")

    def test_bridge_env_uses_discovered_container_reachable_origin_for_local_public_url(self) -> None:
        socket_instance = mock.Mock()
        socket_context = mock.Mock()
        socket_context.__enter__ = mock.Mock(return_value=socket_instance)
        socket_context.__exit__ = mock.Mock(return_value=False)
        socket_instance.getsockname.return_value = ("172.18.0.3", 54321)
        socket_instance.getpeername.return_value = ("172.18.0.2", 8090)

        with (
            mock.patch.object(settings, "external_base_url", "http://localhost:8000"),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://runtime:8090"),
            ),
            mock.patch("ragtime.userspace.runtime_service.socket.socket", return_value=socket_context) as socket_ctor,
        ):
            env = self.service._build_runtime_bridge_env("ws-1", "sess-1")

        socket_ctor.assert_called_once_with(mock.ANY, mock.ANY)
        socket_instance.connect.assert_called_once_with(("runtime", 8090))
        self.assertEqual(
            env["RAGTIME_BRIDGE_URL"],
            "http://172.18.0.3:8000/indexes/userspace/runtime-bridge",
        )

    def test_bridge_env_falls_back_to_public_origin_when_socket_discovery_fails(self) -> None:
        socket_instance = mock.Mock()
        socket_context = mock.Mock()
        socket_context.__enter__ = mock.Mock(return_value=socket_instance)
        socket_context.__exit__ = mock.Mock(return_value=False)
        socket_instance.connect.side_effect = OSError("boom")

        with (
            mock.patch.object(settings, "external_base_url", "http://localhost:8000"),
            mock.patch(
                "ragtime.userspace.runtime_service.get_runtime_manager_request_config",
                return_value=SimpleNamespace(base_url="http://runtime:8090"),
            ),
            mock.patch("ragtime.userspace.runtime_service.socket.socket", return_value=socket_context),
        ):
            env = self.service._build_runtime_bridge_env("ws-1", "sess-1")

        self.assertEqual(
            env["RAGTIME_BRIDGE_URL"],
            "http://localhost:8000/indexes/userspace/runtime-bridge",
        )

    def test_discover_runtime_bridge_host_accepts_private_ipv4_peer(self) -> None:
        socket_instance = mock.Mock()
        socket_context = mock.Mock()
        socket_context.__enter__ = mock.Mock(return_value=socket_instance)
        socket_context.__exit__ = mock.Mock(return_value=False)
        socket_instance.getsockname.return_value = ("172.18.0.3", 54321)
        socket_instance.getpeername.return_value = ("172.18.0.2", 8090)

        with mock.patch("ragtime.userspace.runtime_service.socket.socket", return_value=socket_context):
            host = self.service._discover_runtime_bridge_host("runtime", 8090)

        self.assertEqual(host, "172.18.0.3")

    def test_discover_runtime_bridge_host_rejects_public_ipv4_peer(self) -> None:
        socket_instance = mock.Mock()
        socket_context = mock.Mock()
        socket_context.__enter__ = mock.Mock(return_value=socket_instance)
        socket_context.__exit__ = mock.Mock(return_value=False)
        socket_instance.getsockname.return_value = ("10.0.0.5", 54321)
        socket_instance.getpeername.return_value = ("8.8.8.8", 8090)

        with mock.patch("ragtime.userspace.runtime_service.socket.socket", return_value=socket_context):
            host = self.service._discover_runtime_bridge_host("8.8.8.8", 8090)

        self.assertIsNone(host)

    def test_discover_runtime_bridge_host_accepts_private_ipv6_peer(self) -> None:
        socket_instance = mock.Mock()
        socket_context = mock.Mock()
        socket_context.__enter__ = mock.Mock(return_value=socket_instance)
        socket_context.__exit__ = mock.Mock(return_value=False)
        socket_instance.getsockname.return_value = ("fd00::10", 54321, 0, 0)
        socket_instance.getpeername.return_value = ("fe80::1", 8090, 0, 0)

        with mock.patch("ragtime.userspace.runtime_service.socket.socket", return_value=socket_context):
            host = self.service._discover_runtime_bridge_host("fd00::1", 8090)

        self.assertEqual(host, "fd00::10")

    def test_settings_runtime_bridge_base_url_defaults_blank(self) -> None:
        self.assertEqual(settings.runtime_bridge_base_url, "")

    async def test_provider_start_merges_bridge_env_without_clobbering_user_env(self) -> None:
        with (
            mock.patch.object(self.service, "_require_runtime_manager") as require_manager,
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.get_workspace_runtime_environment",
                mock.AsyncMock(
                    return_value={
                        "MY_VAR": "x",
                        "RAGTIME_BRIDGE_URL": "https://user.invalid/bridge",
                        "RAGTIME_BRIDGE_TOKEN": "user-token",
                    }
                ),
            ),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.get_workspace_runtime_environment_visibility",
                mock.AsyncMock(return_value={"MY_VAR": True}),
            ),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.resolve_workspace_mounts_for_runtime",
                mock.AsyncMock(return_value=[]),
            ),
            mock.patch.object(
                self.service,
                "_runtime_manager_request",
                mock.AsyncMock(return_value={"provider_session_id": "provider-1", "state": "running"}),
            ) as runtime_request,
        ):
            response = await self.service._runtime_provider_start_session(
                "ws-1",
                "user-1",
                session_id="sess-1",
            )

        require_manager.assert_called_once_with()
        self.assertEqual(response["provider_session_id"], "provider-1")
        runtime_request_args = runtime_request.await_args
        assert runtime_request_args is not None
        payload = runtime_request_args.kwargs["json_payload"]
        self.assertEqual(payload["workspace_env"]["MY_VAR"], "x")
        self.assertIn("RAGTIME_BRIDGE_URL", payload["workspace_env"])
        self.assertNotEqual(payload["workspace_env"]["RAGTIME_BRIDGE_URL"], "https://user.invalid/bridge")
        claims = _decode(payload["workspace_env"]["RAGTIME_BRIDGE_TOKEN"])
        self.assertEqual(claims["workspace_id"], "ws-1")
        self.assertEqual(claims["session_id"], "sess-1")

    async def test_bridge_token_excluded_from_env_visibility_listing(self) -> None:
        with (
            mock.patch.object(self.service, "_require_runtime_manager"),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.get_workspace_runtime_environment",
                mock.AsyncMock(return_value={"MY_VAR": "x"}),
            ),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.get_workspace_runtime_environment_visibility",
                mock.AsyncMock(return_value={"MY_VAR": True}),
            ),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.resolve_workspace_mounts_for_runtime",
                mock.AsyncMock(return_value=[]),
            ),
            mock.patch.object(
                self.service,
                "_runtime_manager_request",
                mock.AsyncMock(return_value={"provider_session_id": "provider-1", "state": "running"}),
            ) as runtime_request,
        ):
            await self.service._runtime_provider_start_session(
                "ws-1",
                "user-1",
                session_id="sess-1",
            )

        runtime_request_args = runtime_request.await_args
        assert runtime_request_args is not None
        visibility = runtime_request_args.kwargs["json_payload"]["workspace_env_visibility"]
        self.assertEqual(visibility, {"MY_VAR": True})
        self.assertNotIn("RAGTIME_BRIDGE_TOKEN", visibility)
        self.assertNotIn("RAGTIME_BRIDGE_URL", visibility)

    async def test_refresh_runtime_env_vars_preserves_bridge_env_and_session_id(self) -> None:
        active_row = mock.Mock()
        active_row.id = "sess-1"
        active_row.workspaceId = "ws-1"
        active_row.leasedByUserId = "user-1"
        active_row.state = "running"
        active_row.runtimeProvider = "microvm_pool_v1"
        active_row.providerSessionId = "provider-1"
        active_row.previewInternalUrl = "http://preview"
        active_row.launchFramework = None
        active_row.launchCommand = None
        active_row.launchCwd = None
        active_row.launchPort = None
        active_row.createdAt = datetime.now(UTC)
        active_row.updatedAt = datetime.now(UTC)
        active_row.lastHeartbeatAt = None
        active_row.idleExpiresAt = None
        active_row.ttlExpiresAt = None
        active_row.lastError = None

        with (
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=active_row),
            ),
            mock.patch.object(
                self.service,
                "_ensure_session_row",
                mock.AsyncMock(return_value=self.service._to_runtime_session(active_row)),
            ),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.get_workspace_runtime_environment",
                mock.AsyncMock(
                    return_value={
                        "MY_VAR": "x",
                        "RAGTIME_BRIDGE_URL": "https://user.invalid/bridge",
                        "RAGTIME_BRIDGE_TOKEN": "user-token",
                    }
                ),
            ),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.get_workspace_runtime_environment_visibility",
                mock.AsyncMock(return_value={"MY_VAR": True}),
            ),
            mock.patch.object(
                self.service,
                "_runtime_provider_restart_devserver",
                mock.AsyncMock(),
            ) as restart_devserver,
        ):
            await self.service.refresh_runtime_env_vars("ws-1")

        restart_args = restart_devserver.await_args
        assert restart_args is not None
        restart_kwargs = restart_args.kwargs
        self.assertEqual(restart_args.args[0], "provider-1")
        self.assertEqual(restart_kwargs["workspace_env"]["MY_VAR"], "x")
        self.assertNotEqual(
            restart_kwargs["workspace_env"]["RAGTIME_BRIDGE_URL"],
            "https://user.invalid/bridge",
        )
        claims = _decode(restart_kwargs["workspace_env"]["RAGTIME_BRIDGE_TOKEN"])
        self.assertEqual(claims["workspace_id"], "ws-1")
        self.assertEqual(claims["session_id"], "sess-1")
