#!/usr/bin/env python3
"""Strict, reversible Prisma reconciliation for the local worktree switcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psycopg2
import sqlglot
from sqlglot import exp

ADVISORY_LOCK = 0x72616774696D65


class Refusal(Exception):
    """An expected safety refusal."""


@dataclass(frozen=True)
class Migration:
    name: str
    sha256: str
    path: str


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def migrations(prisma_dir: Path) -> list[Migration]:
    root = prisma_dir / "migrations"
    if not (prisma_dir / "schema.prisma").is_file() or not root.is_dir():
        raise Refusal(f"invalid Prisma package: {prisma_dir}")
    result = [
        Migration(child.name, digest((child / "migration.sql").read_bytes()), str(child / "migration.sql"))
        for child in root.iterdir()
        if child.is_dir() and (child / "migration.sql").is_file()
    ]
    return sorted(result, key=lambda item: item.name)


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_registry(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("version") != 1
        or not isinstance(data.get("legacy_applied"), list)
        or not isinstance(data.get("down_migrations"), list)
        or not isinstance(data.get("baseline_aliases"), list)
    ):
        raise Refusal("unsupported down-migration registry")
    identities: set[tuple[str, str]] = set()
    for item in data["legacy_applied"] + data["down_migrations"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not _valid_digest(item.get("migration_sha256")):
            raise Refusal("registry has invalid migration identity")
        identity = (item["name"], item["migration_sha256"])
        if identity in identities:
            raise Refusal("registry contains duplicate migration identity")
        identities.add(identity)
    _baseline_alias_map(data["baseline_aliases"], identities)
    for item in data["down_migrations"]:
        if not isinstance(item.get("down_sql"), str) or not _valid_digest(item.get("down_sha256")):
            raise Refusal(f"registry has invalid down migration: {item['name']}")
        if digest(item["down_sql"].encode()) != item["down_sha256"]:
            raise Refusal(f"registry down SQL digest does not match: {item['name']}")
        validate_down_sql(item["down_sql"])
    return data


def _baseline_alias_map(items: list[object], legacy: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    aliases: dict[tuple[str, str], str] = {}
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("applied_checksum"), str)
            or not item["applied_checksum"]
            or not _valid_digest(item.get("migration_sha256"))
        ):
            raise Refusal("registry has invalid baseline alias")
        raw_identity = (item["name"], item["applied_checksum"])
        current_identity = (item["name"], item["migration_sha256"])
        if raw_identity in aliases:
            raise Refusal("registry contains duplicate baseline alias")
        if current_identity in legacy:
            raise Refusal("baseline alias conflicts with legacy migration identity")
        aliases[raw_identity] = item["migration_sha256"]
    return aliases


def validate_down_sql(sql: str) -> None:
    """Allow only static reversal statements that fit inside one transaction."""
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except Exception as exc:
        raise Refusal(f"invalid down SQL: {exc}") from exc
    statements = [statement for statement in statements if statement is not None]
    if not statements:
        raise Refusal("down SQL is empty")
    allowed = (exp.Alter, exp.Drop, exp.Delete, exp.TruncateTable)
    for statement in statements:
        if not isinstance(statement, allowed) or statement.find(exp.Command):
            raise Refusal(f"unsupported down SQL statement: {statement.key}")


def read_active_history(connection: Any, legacy: set[tuple[str, str]]) -> list[dict[str, str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT id, migration_name, checksum FROM "_prisma_migrations"
               WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL
               ORDER BY started_at ASC, id ASC"""
        )
        rows = [dict(zip(("id", "name", "sha256"), row)) for row in cursor.fetchall()]
        cursor.execute(
            """SELECT migration_name FROM "_prisma_migrations"
               WHERE finished_at IS NULL AND rolled_back_at IS NULL"""
        )
        unfinished = cursor.fetchall()
    if unfinished:
        raise Refusal("database contains unfinished Prisma migration records")
    filtered = [row for row in rows if (row["name"], row["sha256"]) not in legacy]
    identities = [(row["name"], row["sha256"]) for row in filtered]
    if len(identities) != len(set(identities)) or len({row["name"] for row in filtered}) != len(filtered):
        raise Refusal("database contains duplicate successful migration history")
    return filtered


def normalize_history(rows: list[dict[str, str]], aliases: dict[tuple[str, str], str]) -> list[dict[str, str]]:
    """Compare pinned historical checksums as their current migration identities."""
    return [{"name": row["name"], "sha256": aliases.get((row["name"], row["sha256"]), row["sha256"])} for row in sorted(rows, key=lambda item: item["name"])]


def common_prefix(
    applied: list[dict[str, str]], target: list[Migration], aliases: dict[tuple[str, str], str] | None = None
) -> tuple[list[str], list[dict[str, str]], list[Migration]]:
    aliases = aliases or {}
    applied = sorted(applied, key=lambda item: item["name"])
    count = 0
    while count < len(applied) and count < len(target):
        if (applied[count]["name"], aliases.get((applied[count]["name"], applied[count]["sha256"]), applied[count]["sha256"])) != (
            target[count].name,
            target[count].sha256,
        ):
            break
        count += 1
    return [item.name for item in target[:count]], applied[count:], target[count:]


def _assert_active_aliases_available(
    applied: list[dict[str, str]], aliases: dict[tuple[str, str], str], primary: list[Migration], target: list[Migration]
) -> None:
    primary_ids = {(item.name, item.sha256) for item in primary}
    target_ids = {(item.name, item.sha256) for item in target}
    for row in applied:
        current_sha = aliases.get((row["name"], row["sha256"]))
        if current_sha is None:
            continue
        identity = (row["name"], current_sha)
        if identity not in primary_ids:
            raise Refusal(f"baseline alias is not present in primary: {row['name']}")
        if identity not in target_ids:
            raise Refusal(f"baseline alias is not present in target: {row['name']}")


def _definition_sources(primary: Path, target: Path, sources: list[Path]) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for directory in [primary, target, *sources]:
        for migration in migrations(directory):
            result[(migration.name, migration.sha256)] = Path(migration.path)
    return result


def _cache_path(state: Path, identity: tuple[str, str]) -> Path:
    name, migration_sha = identity
    return state / "cache" / f"{digest(name.encode())}-{migration_sha}.json"


def _load_cached_definition(state: Path, identity: tuple[str, str]) -> tuple[str, str] | None:
    path = _cache_path(state, identity)
    if not path.exists():
        return None
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"corrupt cached migration definition: {identity[0]}") from exc
    required = {"version", "name", "migration_sha256", "migration_sql", "down_sha256", "down_sql"}
    if set(item) != required or item.get("version") != 1:
        raise Refusal(f"corrupt cached migration definition: {identity[0]}")
    if (item["name"], item["migration_sha256"]) != identity or not _valid_digest(item["down_sha256"]):
        raise Refusal(f"corrupt cached migration definition: {identity[0]}")
    if not isinstance(item["migration_sql"], str) or digest(item["migration_sql"].encode()) != identity[1]:
        raise Refusal(f"corrupt cached migration definition: {identity[0]}")
    if not isinstance(item["down_sql"], str) or digest(item["down_sql"].encode()) != item["down_sha256"]:
        raise Refusal(f"corrupt cached migration definition: {identity[0]}")
    validate_down_sql(item["down_sql"])
    return item["down_sql"], item["down_sha256"]


def _cache_definition(state: Path, migration: Migration) -> tuple[str, str] | None:
    candidate = Path(migration.path).parent / "down.sql"
    if not candidate.is_file():
        return None
    down_sql = candidate.read_text(encoding="utf-8")
    validate_down_sql(down_sql)
    identity = (migration.name, migration.sha256)
    existing = _load_cached_definition(state, identity)
    if existing is not None:
        if existing != (down_sql, digest(down_sql.encode())):
            raise Refusal(f"cached down SQL conflicts with target definition: {migration.name}")
        return existing
    path = _cache_path(state, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "name": migration.name,
        "migration_sha256": migration.sha256,
        "migration_sql": Path(migration.path).read_text(encoding="utf-8"),
        "down_sha256": digest(down_sql.encode()),
        "down_sql": down_sql,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return down_sql, payload["down_sha256"]


def _down_for(identity: tuple[str, str], state: Path, definitions: dict[tuple[str, str], Path], registry: dict[str, Any]) -> tuple[str, str] | None:
    cached = _load_cached_definition(state, identity)
    if cached is not None:
        return cached
    migration = definitions.get(identity)
    if migration:
        candidate = migration.parent / "down.sql"
        if candidate.is_file():
            sql = candidate.read_text(encoding="utf-8")
            validate_down_sql(sql)
            return sql, digest(sql.encode())
    for item in registry["down_migrations"]:
        if (item["name"], item["migration_sha256"]) == identity:
            return item["down_sql"], item["down_sha256"]
    return None


def inside(root: Path, path: Path) -> Path:
    root, path = root.resolve(), path.resolve()
    if path != root and root not in path.parents:
        raise Refusal(f"path escapes state root: {path}")
    return path


def package_target(target: Path, package: Path) -> list[Migration]:
    package.mkdir(parents=True, exist_ok=False)
    schema = target / "schema.prisma"
    if not schema.is_file():
        raise Refusal(f"invalid Prisma package: {target}")
    sources = [schema]
    lock = target / "migration_lock.toml"
    if lock.is_file():
        sources.append(lock)
    migration_root = target / "migrations"
    if not migration_root.is_dir():
        raise Refusal(f"invalid Prisma package: {target}")
    migration_paths = list(migration_root.rglob("*"))
    if any(path.is_symlink() for path in migration_paths):
        raise Refusal("Prisma migrations may not contain symlinks")
    sources.extend(path for path in migration_paths if path.is_file())
    manifest: dict[str, str] = {}
    for source in sources:
        relative = source.relative_to(target)
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        destination.write_bytes(data)
        manifest[str(relative)] = digest(data)
    directories = ["migrations", *(str(path.relative_to(target)) for path in migration_paths if path.is_dir())]
    directories.sort()
    (package / "manifest.json").write_text(
        json.dumps({"version": 1, "files": manifest, "directories": directories}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return migrations(package)


def build_plan(args: argparse.Namespace, connection: Any) -> dict[str, Any]:
    state = Path(args.state_root).resolve()
    plan_path = inside(state, Path(args.plan))
    if plan_path.exists() or (plan_path.parent / "package").exists() or (plan_path.parent / "down").exists():
        raise Refusal("plan artifacts already exist")
    target, primary = Path(args.target).resolve(), Path(args.primary).resolve()
    registry = load_registry(Path(args.registry))
    legacy = {(item["name"], item["migration_sha256"]) for item in registry["legacy_applied"]}
    applied = read_active_history(connection, legacy)
    aliases = _baseline_alias_map(registry["baseline_aliases"], legacy)
    primary_migrations = migrations(primary)
    target_available = migrations(target)
    _assert_active_aliases_available(applied, aliases, primary_migrations, target_available)
    package = plan_path.parent / "package"
    target_migrations = package_target(target, package)
    for migration in target_migrations:
        _cache_definition(state, migration)
    prefix, outgoing, forward = common_prefix(applied, target_migrations, aliases)
    aliased_outgoing = [row["name"] for row in outgoing if (row["name"], row["sha256"]) in aliases]
    if aliased_outgoing:
        raise Refusal(f"aliased migration would be outgoing: {', '.join(aliased_outgoing)}")
    definitions = _definition_sources(primary, target, [Path(item) for item in args.source])
    cached_down: list[dict[str, str]] = []
    missing: list[str] = []
    for row in reversed(outgoing):
        found = _down_for((row["name"], row["sha256"]), state, definitions, registry)
        if not found:
            missing.append(row["name"])
            continue
        sql, down_sha = found
        destination = plan_path.parent / "down" / f"{row['name']}-{row['sha256']}.sql"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(sql, encoding="utf-8")
        cached_down.append({"name": row["name"], "migration_sha256": row["sha256"], "down_sha256": down_sha, "path": str(destination)})
    primary_ids = {(item.name, item.sha256) for item in migrations(primary)}
    pending_without_down = [item.name for item in forward if not _down_for((item.name, item.sha256), state, definitions, registry)]
    branch_pending_without_down = [item.name for item in forward if (item.name, item.sha256) not in primary_ids and item.name in pending_without_down]
    reasons = [f"outgoing migrations lack down.sql: {', '.join(missing)}"] if missing else []
    if branch_pending_without_down:
        reasons.append("branch-only target migrations lack down.sql: " + ", ".join(branch_pending_without_down))
    warnings = ["pending forward migrations without down.sql: " + ", ".join(pending_without_down)] if pending_without_down else []
    plan = {
        "version": 1,
        "state_root": str(state),
        "initial_history": applied,
        "expected_final_history": [asdict(item) for item in target_migrations],
        "common_prefix": prefix,
        "reverse": [row["name"] for row in reversed(outgoing)],
        "forward": [item.name for item in forward],
        "outgoing": cached_down,
        "package": str(package),
        "manifest": str(package / "manifest.json"),
        "legacy_applied": [list(item) for item in sorted(legacy)],
        "baseline_aliases": [item for item in registry["baseline_aliases"] if (item["name"], item["applied_checksum"]) in aliases],
    }
    response = {
        "status": "ready" if not reasons else "refused",
        "reasons": reasons,
        "warnings": warnings,
        "common_prefix": prefix,
        "reverse": plan["reverse"],
        "forward": plan["forward"],
        "plan": str(plan_path) if not reasons else None,
    }
    if not reasons:
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return response


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        plan.get("version") != 1
        or not isinstance(plan.get("state_root"), str)
        or not isinstance(plan.get("package"), str)
        or not isinstance(plan.get("manifest"), str)
        or not isinstance(plan.get("outgoing"), list)
        or not isinstance(plan.get("baseline_aliases"), list)
    ):
        raise Refusal("unsupported plan")
    root = Path(plan["state_root"])
    inside(root, path)
    inside(root, Path(plan["package"]))
    inside(root, Path(plan.get("manifest", "")))
    for item in plan.get("expected_final_history", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise Refusal("unsupported plan")
        inside(root, Path(item["path"]))
    for item in plan["outgoing"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise Refusal("unsupported plan")
        inside(root, Path(item["path"]))
    aliases = _baseline_alias_map(plan["baseline_aliases"], {tuple(item) for item in plan.get("legacy_applied", [])})
    expected = {(item["name"], item["sha256"]) for item in plan["expected_final_history"]}
    if any((name, current_sha) not in expected for (name, _), current_sha in aliases.items()):
        raise Refusal("plan baseline alias is not in packaged target")
    return plan


def _assert_package_integrity(plan: dict[str, Any]) -> None:
    try:
        manifest = json.loads(Path(plan["manifest"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal("packaged migration manifest is corrupt") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("files"), dict)
        or not isinstance(manifest.get("directories"), list)
    ):
        raise Refusal("packaged migration manifest is corrupt")
    package = Path(plan["package"])
    actual_paths = {str(path.relative_to(package)): digest(path.read_bytes()) for path in package.rglob("*") if path.is_file() and path.name != "manifest.json"}
    actual_directories = sorted(str(path.relative_to(package)) for path in package.rglob("*") if path.is_dir())
    if manifest["files"] != actual_paths or manifest["directories"] != actual_directories:
        raise Refusal("packaged migration files changed")
    for item in plan["expected_final_history"]:
        source = Path(item["path"])
        if not source.is_file() or digest(source.read_bytes()) != item["sha256"]:
            raise Refusal(f"packaged migration changed: {item['name']}")
    for item in plan["outgoing"]:
        source = Path(item["path"])
        if not source.is_file() or digest(source.read_bytes()) != item["down_sha256"]:
            raise Refusal(f"cached down SQL changed: {item['name']}")
        validate_down_sql(source.read_text(encoding="utf-8"))


def _assert_unchanged(plan: dict[str, Any], connection: Any) -> None:
    actual = read_active_history(connection, {tuple(item) for item in plan["legacy_applied"]})
    if actual != plan["initial_history"]:
        raise Refusal("database migration history changed since check")
    _assert_package_integrity(plan)


def _run_deploy(schema: Path) -> None:
    subprocess.run([sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(schema)], check=True)


def apply(plan_path: Path) -> None:
    plan = _load_plan(plan_path)
    with psycopg2.connect(os.environ["DATABASE_URL"]) as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK,))
            _assert_unchanged(plan, connection)
            try:
                with connection.cursor() as cursor:
                    for item in plan["outgoing"]:
                        cursor.execute(Path(item["path"]).read_text(encoding="utf-8"))
                        cursor.execute(
                            'DELETE FROM "_prisma_migrations" WHERE migration_name = %s AND checksum = %s AND rolled_back_at IS NULL',
                            (item["name"], item["migration_sha256"]),
                        )
                        if cursor.rowcount != 1:
                            raise Refusal(f"Prisma history changed while reversing {item['name']}")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            _run_deploy(Path(plan["package"]) / "schema.prisma")
            verify(plan_path)
        finally:
            connection.rollback()
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK,))


def verify(plan_path: Path) -> None:
    plan = _load_plan(plan_path)
    _assert_package_integrity(plan)
    with psycopg2.connect(os.environ["DATABASE_URL"]) as connection:
        actual = read_active_history(connection, {tuple(item) for item in plan["legacy_applied"]})
    expected = [{"name": item["name"], "sha256": item["sha256"]} for item in plan["expected_final_history"]]
    aliases = _baseline_alias_map(plan["baseline_aliases"], {tuple(item) for item in plan["legacy_applied"]})
    if normalize_history(actual, aliases) != expected:
        raise Refusal("active Prisma history does not exactly match packaged target")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    for flag in ("target", "primary", "state_root", "plan", "registry"):
        check.add_argument("--" + flag.replace("_", "-"), required=True)
    check.add_argument("--source", action="append", default=[])
    for command in ("apply", "verify"):
        item = sub.add_parser(command)
        item.add_argument("--plan", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            with psycopg2.connect(os.environ["DATABASE_URL"]) as connection:
                response = build_plan(args, connection)
            print(json.dumps(response, sort_keys=True))
            return 0 if response["status"] == "ready" else 2
        if args.command == "apply":
            apply(Path(args.plan))
        else:
            verify(Path(args.plan))
        return 0
    except Refusal as exc:
        if args.command == "check":
            print(
                json.dumps(
                    {"status": "refused", "reasons": [str(exc)], "warnings": [], "common_prefix": [], "reverse": [], "forward": [], "plan": None},
                    sort_keys=True,
                )
            )
        else:
            print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"worktree migration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
