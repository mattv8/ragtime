import pytest

from ragtime.core.docker_ssh import docker_ssh_config_from_dict, execute_docker_command_on_remote_host
from ragtime.core.ssh import SSHConfig, SSHResult
from ragtime.indexer.routes import (
    DockerDiscoverRequest,
    _discover_remote_docker_resources,
    _heartbeat_mysql,
    _heartbeat_odoo,
    _heartbeat_postgres,
    _test_odoo_docker_connection,
    _test_postgres_connection,
)
from ragtime.indexer.tool_health import get_heartbeat_timeout_seconds
from ragtime.tools.odoo_shell import ODOO_OUTPUT_BEGIN_MARKER, ODOO_OUTPUT_END_MARKER, filter_odoo_output


def test_docker_ssh_config_from_dict_respects_disabled_toggle() -> None:
    config = docker_ssh_config_from_dict(
        {
            "docker_ssh_enabled": False,
            "docker_ssh_host": "example.com",
            "docker_ssh_user": "ubuntu",
            "docker_ssh_password": "secret",
        }
    )

    assert config is None


def test_docker_ssh_config_from_dict_maps_prefixed_fields() -> None:
    config = docker_ssh_config_from_dict(
        {
            "docker_ssh_enabled": True,
            "docker_ssh_host": "example.com",
            "docker_ssh_port": 2222,
            "docker_ssh_user": "ubuntu",
            "docker_ssh_key_content": "PRIVATE KEY",
            "docker_ssh_key_passphrase": "passphrase",
        },
        timeout=45,
    )

    assert config is not None
    assert config.host == "example.com"
    assert config.port == 2222
    assert config.user == "ubuntu"
    assert config.key_content == "PRIVATE KEY"
    assert config.key_passphrase == "passphrase"
    assert config.timeout == 45


def test_execute_docker_command_on_remote_host_quotes_command_and_embeds_input_heredoc(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def fake_execute_ssh_command(config: SSHConfig, command: str, input_data: str | None = None) -> SSHResult:
        captured["command"] = command
        captured["input_data"] = input_data
        return SSHResult(stdout="ok", stderr="", exit_code=0, success=True)

    monkeypatch.setattr("ragtime.core.docker_ssh.execute_ssh_command", fake_execute_ssh_command)

    ssh_config = SSHConfig(host="example.com", user="ubuntu", password="secret")
    result = execute_docker_command_on_remote_host(
        ssh_config,
        ["docker", "exec", "-i", "odoo-prod", "bash", "-c", "echo '$POSTGRES_DB'"],
        input_data="print('hello')\n",
    )

    assert result.success is True
    assert captured["command"] == (
        "docker exec -i odoo-prod bash -c 'echo '\"'\"'$POSTGRES_DB'\"'\"'' <<'RAGTIME_DOCKER_STDIN'\nprint('hello')\n\nRAGTIME_DOCKER_STDIN"
    )
    assert captured["input_data"] is None


def test_remote_docker_heartbeat_uses_longer_timeout() -> None:
    assert get_heartbeat_timeout_seconds({"docker_ssh_enabled": True}) == 15.0


@pytest.mark.anyio
async def test_remote_odoo_docker_heartbeat_only_checks_container_running(monkeypatch) -> None:
    captured_commands: list[list[str]] = []

    async def fake_run_remote_docker_command(ssh_config, command, timeout=10.0, input_data=None):
        captured_commands.append(command)
        assert input_data is None
        assert timeout == 10.0
        return SSHResult(stdout="true\n", stderr="", exit_code=0, success=True)

    monkeypatch.setattr("ragtime.indexer.routes._run_remote_docker_command", fake_run_remote_docker_command)

    result = await _heartbeat_odoo(
        {
            "mode": "docker",
            "container": "doploy-dc56c846",
            "docker_ssh_enabled": True,
            "docker_ssh_host": "example.com",
            "docker_ssh_user": "root",
            "docker_ssh_password": "secret",
        }
    )

    assert result.success is True
    assert result.message == "OK"
    assert captured_commands == [["docker", "inspect", "-f", "{{.State.Running}}", "doploy-dc56c846"]]


@pytest.mark.anyio
async def test_remote_postgres_docker_heartbeat_uses_remote_docker(monkeypatch) -> None:
    captured_commands: list[list[str]] = []

    async def fake_run_remote_docker_command(ssh_config, command, timeout=10.0, input_data=None):
        captured_commands.append(command)
        assert input_data is None
        assert timeout == 10.0
        return SSHResult(stdout=" ?column?\n----------\n        1\n(1 row)\n", stderr="", exit_code=0, success=True)

    monkeypatch.setattr("ragtime.indexer.routes._run_remote_docker_command", fake_run_remote_docker_command)

    result = await _heartbeat_postgres(
        {
            "container": "doploy-dc56c846-db",
            "docker_ssh_enabled": True,
            "docker_ssh_host": "example.com",
            "docker_ssh_user": "root",
            "docker_ssh_password": "secret",
            "ssh_tunnel_enabled": True,
        }
    )

    assert result.success is True
    assert result.message == "OK"
    assert captured_commands == [
        [
            "docker",
            "exec",
            "-i",
            "doploy-dc56c846-db",
            "bash",
            "-c",
            'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1;"',
        ]
    ]


@pytest.mark.anyio
async def test_remote_mysql_docker_heartbeat_prefers_container_over_ssh_tunnel(monkeypatch) -> None:
    captured_commands: list[list[str]] = []

    async def fake_run_remote_docker_command(ssh_config, command, timeout=10.0, input_data=None):
        captured_commands.append(command)
        assert input_data is None
        assert timeout == 10.0
        command_text = " ".join(command)
        if command_text == "docker exec mysql-prod printenv MYSQL_USER":
            return SSHResult(stdout="mysql_user\n", stderr="", exit_code=0, success=True)
        if command_text == "docker exec mysql-prod printenv MYSQL_ROOT_PASSWORD":
            return SSHResult(stdout="root-secret\n", stderr="", exit_code=0, success=True)
        if command_text == "docker exec mysql-prod printenv MYSQL_PASSWORD":
            return SSHResult(stdout="user-secret\n", stderr="", exit_code=0, success=True)
        if command_text == "docker exec mysql-prod printenv MYSQL_DATABASE":
            return SSHResult(stdout="app_db\n", stderr="", exit_code=0, success=True)
        if command_text == "docker exec mysql-prod mysql -umysql_user -puser-secret -N -e SELECT 1 app_db":
            return SSHResult(stdout="1\n", stderr="", exit_code=0, success=True)
        raise AssertionError(f"Unexpected command: {command_text}")

    monkeypatch.setattr("ragtime.indexer.routes._run_remote_docker_command", fake_run_remote_docker_command)

    result = await _heartbeat_mysql(
        {
            "container": "mysql-prod",
            "docker_ssh_enabled": True,
            "docker_ssh_host": "example.com",
            "docker_ssh_user": "root",
            "docker_ssh_password": "secret",
            "ssh_tunnel_enabled": True,
        }
    )

    assert result.success is True
    assert result.message == "OK"
    assert captured_commands == [
        ["docker", "exec", "mysql-prod", "printenv", "MYSQL_USER"],
        ["docker", "exec", "mysql-prod", "printenv", "MYSQL_ROOT_PASSWORD"],
        ["docker", "exec", "mysql-prod", "printenv", "MYSQL_PASSWORD"],
        ["docker", "exec", "mysql-prod", "printenv", "MYSQL_DATABASE"],
        ["docker", "exec", "mysql-prod", "mysql", "-umysql_user", "-puser-secret", "-N", "-e", "SELECT 1", "app_db"],
    ]


@pytest.mark.anyio
async def test_remote_docker_discovery_falls_back_to_network_inspect(monkeypatch) -> None:
    async def fake_run_remote_docker_command(ssh_config, command, timeout=10.0, input_data=None):
        command_text = " ".join(command)
        if command_text == "docker version --format {{.Server.Version}}":
            return SSHResult(stdout="27.0.0", stderr="", exit_code=0, success=True)
        if command_text == "docker network ls --format {{json .}}":
            return SSHResult(
                stdout='{"Name":"doploy-dc56c846-network","Driver":"bridge","Scope":"local"}\n',
                stderr="",
                exit_code=0,
                success=True,
            )
        if command_text == "docker ps --format {{json .}}":
            return SSHResult(stdout="", stderr="", exit_code=0, success=True)
        if command_text == "docker network inspect doploy-dc56c846-network --format {{json .Containers}}":
            return SSHResult(
                stdout='{"abc":{"Name":"odoo-web"}}',
                stderr="",
                exit_code=0,
                success=True,
            )
        if command_text.startswith("docker inspect -f") and command[-1] == "odoo-web":
            return SSHResult(
                stdout="/odoo-web|odoo:17|running|doploy-dc56c846-network \n",
                stderr="",
                exit_code=0,
                success=True,
            )
        raise AssertionError(f"Unexpected command: {command_text}")

    monkeypatch.setattr("ragtime.indexer.routes._run_remote_docker_command", fake_run_remote_docker_command)

    result = await _discover_remote_docker_resources(
        DockerDiscoverRequest(
            docker_ssh_host="example.com",
            docker_ssh_user="ubuntu",
            docker_ssh_password="secret",
        )
    )

    assert result.success is True
    assert [container.name for container in result.containers] == ["odoo-web"]
    assert result.containers[0].has_odoo is True
    assert result.networks[0].containers == ["odoo-web"]


@pytest.mark.anyio
async def test_remote_odoo_docker_connection_does_not_initialize_shell(monkeypatch) -> None:
    captured_commands: list[list[str]] = []

    async def fake_run_remote_docker_command(ssh_config, command, timeout=10.0, input_data=None):
        captured_commands.append(command)
        assert input_data is None
        command_text = " ".join(command)
        if command_text == "docker inspect -f {{.State.Running}} doploy-dc56c846":
            return SSHResult(stdout="true\n", stderr="", exit_code=0, success=True)
        if command_text == "docker exec -i doploy-dc56c846 odoo --version":
            return SSHResult(stdout="Odoo Server 17.0\n", stderr="", exit_code=0, success=True)
        raise AssertionError(f"Unexpected command: {command_text}")

    monkeypatch.setattr("ragtime.indexer.routes._run_remote_docker_command", fake_run_remote_docker_command)

    result = await _test_odoo_docker_connection(
        {
            "container": "doploy-dc56c846",
            "database": "hammerton_production",
            "docker_ssh_enabled": True,
            "docker_ssh_host": "example.com",
            "docker_ssh_user": "root",
            "docker_ssh_password": "secret",
        }
    )

    assert result.success is True
    assert result.message.startswith("Odoo command accessible on remote Docker host")
    assert captured_commands == [
        ["docker", "inspect", "-f", "{{.State.Running}}", "doploy-dc56c846"],
        ["docker", "exec", "-i", "doploy-dc56c846", "odoo", "--version"],
    ]


@pytest.mark.anyio
async def test_remote_postgres_docker_connection_prefers_container_over_ssh_tunnel(monkeypatch) -> None:
    captured_commands: list[list[str]] = []

    async def fake_run_remote_docker_command(ssh_config, command, timeout=10.0, input_data=None):
        captured_commands.append(command)
        assert input_data is None
        assert timeout == 35.0
        return SSHResult(stdout=" ?column?\n----------\n        1\n(1 row)\n", stderr="", exit_code=0, success=True)

    monkeypatch.setattr("ragtime.indexer.routes._run_remote_docker_command", fake_run_remote_docker_command)

    result = await _test_postgres_connection(
        {
            "container": "doploy-dc56c846-db",
            "docker_ssh_enabled": True,
            "docker_ssh_host": "example.com",
            "docker_ssh_user": "root",
            "docker_ssh_password": "secret",
            "ssh_tunnel_enabled": True,
            "ssh_tunnel_host": "example.com",
        }
    )

    assert result.success is True
    assert result.message == "PostgreSQL connection successful (remote Docker)"
    assert result.details and result.details["mode"] == "docker_ssh"
    assert captured_commands == [
        [
            "docker",
            "exec",
            "-i",
            "doploy-dc56c846-db",
            "bash",
            "-c",
            'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1;"',
        ]
    ]


def test_filter_odoo_output_prefers_marker_delimited_payload() -> None:
    output = (
        "2026-06-08 21:23:43,176 INFO hammerton_production odoo.modules.loading: Modules loaded.\n"
        f"{ODOO_OUTPUT_BEGIN_MARKER}\n"
        "real user output\n"
        f"{ODOO_OUTPUT_END_MARKER}\n"
        "2026-06-08 21:23:43,195 INFO hammerton_production odoo.modules.registry: Registry loaded in 4.225s\n"
    )

    assert filter_odoo_output(output) == "real user output"
