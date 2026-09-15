import unittest
from unittest import mock

import pymssql

from ragtime.core.ssh import SSHTunnelConfig
from ragtime.tools import mssql


class _Cursor:
    def __init__(self, as_dict: bool) -> None:
        self.as_dict = as_dict
        self.query = ""
        self.closed = False

    def execute(self, query: str) -> None:
        self.query = query
        if query == "SELECT @@VERSION" and self.as_dict:
            raise pymssql.ColumnsWithoutNamesError([0])

    def fetchone(self) -> tuple[str] | dict[str, str]:
        if self.query == "SELECT @@VERSION":
            return ("Microsoft SQL Server 2019\n15.0.4385.2",)
        if self.query == "SELECT DB_NAME() AS db_name":
            return {"db_name": "PDM"} if self.as_dict else ("PDM",)
        raise AssertionError(f"Unexpected query: {self.query}")

    def close(self) -> None:
        self.closed = True


class _Connection:
    def __init__(self, as_dict: bool = True) -> None:
        self.as_dict = as_dict
        self.cursors: list[_Cursor] = []
        self.closed = False

    def cursor(self, as_dict: bool | None = None) -> _Cursor:
        cursor = _Cursor(self.as_dict if as_dict is None else as_dict)
        self.cursors.append(cursor)
        return cursor

    def close(self) -> None:
        self.closed = True


class _Tunnel:
    def __init__(self, config: SSHTunnelConfig) -> None:
        self.config = config
        self.stopped = False

    def start(self) -> int:
        return 50123

    def stop(self) -> None:
        self.stopped = True


class MssqlConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_connection_uses_tuple_cursor_for_unnamed_version_column(self) -> None:
        connections: list[_Connection] = []

        def connect_driver(**kwargs: object) -> _Connection:
            connection = _Connection(as_dict=kwargs["as_dict"] is True)
            connections.append(connection)
            return connection

        with mock.patch.object(pymssql, "connect", side_effect=connect_driver) as connect:
            success, message, details = await mssql.test_mssql_connection(host="sql.example.test", port=1433, user="reader", password="secret", database="PDM")

        connection = connections[0]
        self.assertTrue(success)
        self.assertEqual(message, "Connected to PDM successfully")
        self.assertEqual(
            details,
            {"version": "Microsoft SQL Server 2019", "database": "PDM", "host": "sql.example.test", "port": 1433},
        )
        connect.assert_called_once_with(
            server="sql.example.test",
            port="1433",
            user="reader",
            password="secret",
            database="PDM",
            login_timeout=10,
            timeout=10,
            as_dict=True,
        )
        self.assertEqual([cursor.as_dict for cursor in connection.cursors], [False])
        self.assertTrue(connection.cursors[0].closed)
        self.assertTrue(connection.closed)

    async def test_ssh_connection_reports_tunnel_metadata_and_cleans_up(self) -> None:
        connections: list[_Connection] = []
        tunnels: list[_Tunnel] = []

        def connect_driver(**kwargs: object) -> _Connection:
            connection = _Connection(as_dict=kwargs["as_dict"] is True)
            connections.append(connection)
            return connection

        def create_tunnel(config: SSHTunnelConfig) -> _Tunnel:
            tunnel = _Tunnel(config)
            tunnels.append(tunnel)
            return tunnel

        tunnel_config = {
            "ssh_tunnel_host": "bastion.example.test",
            "ssh_tunnel_user": "tunnel-user",
            "ssh_tunnel_password": "tunnel-secret",
            "host": "sql.internal",
            "port": 1433,
        }
        with (
            mock.patch.object(pymssql, "connect", side_effect=connect_driver) as connect,
            mock.patch.object(mssql, "SSHTunnel", side_effect=create_tunnel),
        ):
            success, message, details = await mssql.test_mssql_connection(
                host="sql.internal",
                user="reader",
                password="secret",
                database="PDM",
                ssh_tunnel_config=tunnel_config,
            )

        connection = connections[0]
        tunnel = tunnels[0]
        self.assertTrue(success)
        self.assertEqual(message, "Connected to PDM successfully (via SSH tunnel)")
        self.assertEqual(
            details,
            {"version": "Microsoft SQL Server 2019", "database": "PDM", "mode": "ssh_tunnel", "ssh_host": "bastion.example.test"},
        )
        connect.assert_called_once_with(
            server="127.0.0.1",
            port="50123",
            user="reader",
            password="secret",
            database="PDM",
            login_timeout=10,
            timeout=10,
            as_dict=True,
        )
        self.assertTrue(connection.cursors[0].closed)
        self.assertTrue(connection.closed)
        self.assertEqual(len(tunnels), 1)
        self.assertEqual(tunnel.config.ssh_host, "bastion.example.test")
        self.assertEqual(tunnel.config.remote_host, "sql.internal")
        self.assertEqual(tunnel.config.remote_port, 1433)
        self.assertTrue(tunnel.stopped)

    async def test_operational_error_remains_connection_failure(self) -> None:
        with mock.patch.object(pymssql, "connect", side_effect=pymssql.OperationalError("login failed")):
            success, message, details = await mssql.test_mssql_connection(host="sql.example.test", user="reader", password="secret", database="PDM")

        self.assertFalse(success)
        self.assertIn("Login failed for SQL Server user", message)
        self.assertIsNone(details)
