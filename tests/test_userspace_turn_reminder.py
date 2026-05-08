import sys
import types
import unittest
from importlib import util
from pathlib import Path

_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "ragtime" / "rag" / "prompts.py"
_SPEC = util.spec_from_file_location("ragtime_rag_prompts_for_test", _PROMPTS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load prompts module for tests")
_prompts = util.module_from_spec(_SPEC)
_MODULE_STUB_NAMES = [
    "ragtime.core",
    "ragtime.core.entrypoint_status",
    "ragtime.core.user_identity",
]
_original_modules = {name: sys.modules.get(name) for name in _MODULE_STUB_NAMES}
try:
    fake_core_package = types.ModuleType("ragtime.core")
    fake_entrypoint_module = types.ModuleType("ragtime.core.entrypoint_status")
    setattr(fake_entrypoint_module, "EntrypointStatus", object)
    fake_user_identity_module = types.ModuleType("ragtime.core.user_identity")
    setattr(
        fake_user_identity_module,
        "normalize_user_identity",
        lambda username, display_name: (
            (username or "").strip(),
            (display_name or "").strip(),
        ),
    )
    sys.modules["ragtime.core"] = fake_core_package
    sys.modules["ragtime.core.entrypoint_status"] = fake_entrypoint_module
    sys.modules["ragtime.core.user_identity"] = fake_user_identity_module
    _SPEC.loader.exec_module(_prompts)
finally:
    for name, module in _original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

build_userspace_turn_reminder = _prompts.build_userspace_turn_reminder
build_userspace_turn_reminder_with_env_vars = (
    _prompts.build_userspace_turn_reminder_with_env_vars
)


class UserSpaceTurnReminderTests(unittest.TestCase):
    def test_runtime_status_line_is_included(self) -> None:
        runtime_line = (
            "- Current runtime blocker: session_state=running, phase=failed, "
            "devserver_running=false. Last runtime error: Dev server exited with "
            "code 1: SyntaxError: Unexpected token '||' Node.js v20.19.2. "
            "Fix this before unrelated feature work.\n"
        )

        reminder = build_userspace_turn_reminder(
            include_sqlite_persistence=False,
            runtime_status_reminder_line=runtime_line,
        )

        self.assertIn(runtime_line, reminder)
        self.assertLess(
            reminder.index(runtime_line),
            reminder.index("- Prefer incremental edits"),
        )

    def test_runtime_status_line_combines_with_env_var_hint(self) -> None:
        env_line = "- Workspace env vars (keys only): API_KEY(set).\n"
        runtime_line = (
            "- Current runtime blocker: session_state=running, "
            "devserver_running=false. Last runtime error: boom.\n"
        )

        reminder = build_userspace_turn_reminder_with_env_vars(
            include_sqlite_persistence=False,
            env_var_reminder_line=env_line,
            runtime_status_reminder_line=runtime_line,
        )

        self.assertIn(env_line, reminder)
        self.assertIn(runtime_line, reminder)
        self.assertLess(reminder.index(env_line), reminder.index(runtime_line))

    def test_default_reminder_has_no_runtime_placeholder(self) -> None:
        reminder = build_userspace_turn_reminder(include_sqlite_persistence=False)

        self.assertNotIn("runtime_status_reminder_line", reminder)
        self.assertNotIn("Current runtime blocker", reminder)


if __name__ == "__main__":
    unittest.main()