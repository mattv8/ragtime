"""Backward-compatible host shim for chat fixture export and import-dev."""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

# Running a script sets sys.path to scripts/, unlike ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragtime.fixtures import chat
from ragtime.fixtures.cli import main as fixture_main


def export_sql(conversation_id: str) -> str:
    literal_id = "'" + str(uuid.UUID(conversation_id)) + "'"
    return "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;\n" + chat.fixture_select().replace("%s", literal_id) + ";\nCOMMIT;\n"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"export", "import"}:
        raise SystemExit(fixture_main(sys.argv[1:]))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sql = commands.add_parser("export-sql")
    sql.add_argument("--conversation-id", required=True)
    legacy_import = commands.add_parser("import-dev")
    legacy_import.add_argument("file")
    legacy_import.add_argument("--owner", required=True)
    args = parser.parse_args()
    if args.command == "export-sql":
        print(export_sql(args.conversation_id))
        return
    if args.command == "import-dev":
        try:
            document = json.loads(Path(args.file).read_text(encoding="utf-8"))
            title = f"[Branch debug] {document['conversation']['title']}"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            print("Fixture input must be valid UTF-8 JSON.", file=sys.stderr)
            raise SystemExit(1)
        with open(args.file, "rb") as fixture_file:
            result = subprocess.run(
                ["docker", "exec", "-i", "ragtime-dev", "import", "chat", "-", "--owner", args.owner, "--title", title],
                stdin=fixture_file,
                check=False,
            )
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
