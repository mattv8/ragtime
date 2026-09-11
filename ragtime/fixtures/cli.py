"""Administrative fixture import/export command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping

from . import chat
from .common import FixtureError

HANDLERS = {"chat": chat}


def build_parser(handlers: Mapping[str, object] = HANDLERS) -> argparse.ArgumentParser:
    """Build a small explicit registry; fixture types own their argument contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, register_name in (("export", "register_export"), ("import", "register_import")):
        command_parser = commands.add_parser(command)
        types = command_parser.add_subparsers(dest="fixture_type", required=True)
        for fixture_type, handler in handlers.items():
            type_parser = types.add_parser(fixture_type)
            getattr(handler, register_name)(type_parser)
    return parser


def main(argv: list[str] | None = None, *, stdout=None, stderr=None, stdin=None, handlers: Mapping[str, object] = HANDLERS) -> int:
    args = build_parser(handlers).parse_args(argv)
    args.stdout = sys.stdout if stdout is None else stdout
    args.stderr = sys.stderr if stderr is None else stderr
    args.stdin = sys.stdin if stdin is None else stdin
    try:
        return args.run(args)
    except FixtureError as error:
        print(str(error), file=args.stderr)
        return 1
    except Exception:
        print(f"Fixture {args.command} failed.", file=args.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
