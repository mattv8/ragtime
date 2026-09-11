"""Ordered, disk-backed staging for bounded document indexing attempts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class SpoolRecord:
    record_id: str
    source: str
    ordinal: int
    text_path: str
    metadata: dict[str, Any]
    text_bytes: int


@dataclass(frozen=True)
class SpoolBatch:
    records: tuple[SpoolRecord, ...]
    estimated_peak_bytes: int


@dataclass(frozen=True)
class EmbeddedBatch:
    records: tuple[SpoolRecord, ...]
    vectors_path: str
    vector_offset_bytes: int
    rows: int
    dimensions: int


@dataclass(frozen=True)
class SpoolTaskOutput:
    manifest_path: str
    record_count: int
    text_bytes: int
    peak_rss_bytes: int


class IndexingSpool:
    """SQLite ordering journal; task output files are always attempt-relative."""

    MAX_DISK_RESERVE_BYTES = 1024 * 1024 * 1024

    def __init__(self, root: Path, connection: sqlite3.Connection) -> None:
        self.root, self._connection = root.resolve(), connection

    @classmethod
    def open_attempt(cls, root: Path, job_id: str, fingerprint: str) -> "IndexingSpool":
        if not job_id or Path(job_id).name != job_id:
            raise ValueError("Spool job id must be a single path component")
        base = root.resolve()
        base.mkdir(parents=True, exist_ok=True)
        job_root = base / job_id
        job_root.mkdir(exist_ok=True)
        if job_root.resolve().parent != base:
            raise ValueError("Spool job directory escapes its root")
        # Resume the newest valid matching journal. Task output is committed only
        # through SQLite, so files from interrupted workers with no record rows
        # are discarded rather than replayed.
        for attempt in sorted(job_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if attempt.is_symlink() or not attempt.is_dir() or not (attempt / "spool.sqlite3").is_file():
                continue
            try:
                candidate = cls.open_existing(attempt, readonly=False)
                if candidate.get_state("fingerprint") == fingerprint:
                    candidate._reconcile_task_files()
                    return candidate
                candidate.close()
            except (OSError, sqlite3.Error):
                continue
        attempt = job_root / uuid.uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        # Pipeline calls are serialized by its parent, but writes are offloaded
        # from the event loop.  The connection is intentionally single-owner,
        # not one connection per worker (workers only create task files).
        conn = sqlite3.connect(attempt / "spool.sqlite3", check_same_thread=False)
        conn.execute("CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE records (stage TEXT NOT NULL, position INTEGER NOT NULL, record_id TEXT NOT NULL, source TEXT NOT NULL, ordinal INTEGER NOT NULL, text_path TEXT NOT NULL, metadata TEXT NOT NULL, text_bytes INTEGER NOT NULL, PRIMARY KEY(stage, position), UNIQUE(stage, record_id))"
        )
        conn.execute(
            "CREATE TABLE embeddings (position INTEGER PRIMARY KEY, record_id TEXT NOT NULL UNIQUE, vectors_path TEXT NOT NULL, vector_offset INTEGER NOT NULL, dimensions INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO state VALUES ('fingerprint', ?)", (json.dumps(fingerprint),))
        conn.commit()
        return cls(attempt, conn)

    @classmethod
    def open_existing(cls, root: Path, *, readonly: bool = True) -> "IndexingSpool":
        path = root.resolve()
        db = path / "spool.sqlite3"
        if not path.is_dir() or not db.is_file():
            raise FileNotFoundError(f"Indexing spool attempt is missing: {path}")
        uri = f"file:{db}?mode=ro" if readonly else str(db)
        return cls(path, sqlite3.connect(uri, uri=readonly, check_same_thread=False))

    def _relative(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Spool paths must be attempt-relative")
        return path

    def _file(self, value: str) -> Path:
        """Resolve an attempt-relative file without following an escaping symlink."""
        path = self.root / self._relative(value)
        try:
            path.resolve(strict=True).relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Spool path escapes its attempt through a symlink") from exc
        return path

    @classmethod
    def ensure_path_disk_space(cls, root: Path, additional_bytes: int) -> None:
        """Keep a reserve so a partial spool never exhausts the index volume."""
        if additional_bytes < 0:
            raise ValueError("Spool disk reservation cannot be negative")
        volume_path = root
        while not volume_path.exists() and volume_path != volume_path.parent:
            volume_path = volume_path.parent
        usage = shutil.disk_usage(volume_path)
        reserve = min(cls.MAX_DISK_RESERVE_BYTES, usage.total // 20)
        if usage.free - additional_bytes < reserve:
            raise OSError("Insufficient disk space to stage index output safely")

    def ensure_disk_space(self, additional_bytes: int) -> None:
        self.ensure_path_disk_space(self.root, additional_bytes)

    @staticmethod
    def stable_id(source: str, ordinal: int, output_ordinal: int, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{source}\0{ordinal}\0{output_ordinal}\0{digest}".encode()).hexdigest()

    def _accept(self, output: SpoolTaskOutput, stage: str) -> None:
        manifest = self._file(output.manifest_path)
        count = 0
        text_bytes = 0
        position = self._connection.execute("SELECT COALESCE(MAX(position) + 1, 0) FROM records WHERE stage = ?", (stage,)).fetchone()[0]
        try:
            with self._connection:
                with manifest.open(encoding="utf-8") as lines:
                    for line in lines:
                        data = json.loads(line)
                        record = SpoolRecord(**data)
                        text_path = self._relative(record.text_path)
                        text_file = self._file(record.text_path)
                        if not text_file.is_file():
                            raise FileNotFoundError(f"Spool text record is missing: {text_path}")
                        content = text_file.read_bytes()
                        if len(content) != record.text_bytes:
                            raise ValueError("Spool text record byte count does not match its descriptor")
                        existing = self._connection.execute(
                            "SELECT source, ordinal, text_path, metadata, text_bytes FROM records WHERE stage=? AND record_id=?",
                            (stage, record.record_id),
                        ).fetchone()
                        values = (record.source, record.ordinal, str(text_path), json.dumps(record.metadata, sort_keys=True), record.text_bytes)
                        if existing is not None:
                            existing_content = self._file(existing[2]).read_bytes()
                            # Task directories are attempt-relative and random on
                            # replay. Compare durable identity, not that path.
                            if (
                                existing[0] != record.source
                                or existing[1] != record.ordinal
                                or existing[3] != values[3]
                                or existing[4] != record.text_bytes
                                or hashlib.sha256(existing_content).digest() != hashlib.sha256(content).digest()
                            ):
                                raise ValueError("Spool task record conflicts with committed output")
                        else:
                            self._connection.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (stage, position, record.record_id, *values))
                            position += 1
                        count += 1
                        text_bytes += record.text_bytes
                if count != output.record_count or text_bytes != output.text_bytes:
                    raise ValueError("Spool task manifest count or byte total does not match descriptor")
        except Exception:
            self._connection.rollback()
            raise

    def accept_documents(self, output: SpoolTaskOutput) -> None:
        self._accept(output, "documents")

    def accept_chunks(self, output: SpoolTaskOutput) -> None:
        self._accept(output, "chunks")

    def _iter_records(self, stage: str, max_documents: int, max_text_bytes: int) -> Iterator[SpoolBatch]:
        rows = self._connection.execute(
            "SELECT record_id, source, ordinal, text_path, metadata, text_bytes FROM records WHERE stage=? ORDER BY position", (stage,)
        )
        batch: list[SpoolRecord] = []
        total = 0
        for row in rows:
            record = SpoolRecord(row[0], row[1], row[2], row[3], json.loads(row[4]), row[5])
            if batch and (len(batch) >= max_documents or total + record.text_bytes > max_text_bytes):
                yield SpoolBatch(tuple(batch), total)
                batch, total = [], 0
            batch.append(record)
            total += record.text_bytes
        if batch:
            yield SpoolBatch(tuple(batch), total)

    def iter_documents(self, *, max_documents: int, max_text_bytes: int) -> Iterator[SpoolBatch]:
        return self._iter_records("documents", max_documents, max_text_bytes)

    def iter_chunks(self, *, max_documents: int, max_text_bytes: int) -> Iterator[SpoolBatch]:
        return self._iter_records("chunks", max_documents, max_text_bytes)

    def batch_after(self, stage: str, position: int, *, max_documents: int, max_text_bytes: int) -> tuple[SpoolBatch | None, int]:
        """Read one bounded batch without retaining a SQLite cursor across awaits."""
        rows = self._connection.execute(
            "SELECT position, record_id, source, ordinal, text_path, metadata, text_bytes FROM records WHERE stage=? AND position>=? ORDER BY position",
            (stage, position),
        )
        records: list[SpoolRecord] = []
        total = 0
        next_position = position
        for row in rows:
            if records and (len(records) >= max_documents or total + row[6] > max_text_bytes):
                break
            records.append(SpoolRecord(row[1], row[2], row[3], row[4], json.loads(row[5]), row[6]))
            total += row[6]
            next_position = row[0] + 1
        return (SpoolBatch(tuple(records), total) if records else None, next_position)

    def unembedded_chunk_batch_after(self, position: int, *, max_documents: int, max_text_bytes: int) -> tuple[SpoolBatch | None, int]:
        """Read the next chunks lacking vector rows, advancing past committed rows."""
        rows = self._connection.execute(
            """SELECT r.position, r.record_id, r.source, r.ordinal, r.text_path, r.metadata, r.text_bytes
               FROM records r LEFT JOIN embeddings e ON e.record_id=r.record_id
               WHERE r.stage='chunks' AND r.position>=? AND e.record_id IS NULL ORDER BY r.position""",
            (position,),
        )
        records: list[SpoolRecord] = []
        total = 0
        next_position = position
        for row in rows:
            if records and (len(records) >= max_documents or total + row[6] > max_text_bytes):
                break
            records.append(SpoolRecord(row[1], row[2], row[3], row[4], json.loads(row[5]), row[6]))
            total += row[6]
            next_position = row[0] + 1
        return (SpoolBatch(tuple(records), total) if records else None, next_position)

    def document_ordinals(self) -> set[int]:
        return {row[0] for row in self._connection.execute("SELECT DISTINCT ordinal FROM records WHERE stage='documents'")}

    def replace_chunks(self, originals: tuple[SpoolRecord, ...], replacements: list[list[Any]]) -> tuple[SpoolRecord, ...]:
        """Atomically substitute unembedded chunks after a context-length failure."""
        if len(originals) != len(replacements):
            raise ValueError("Chunk replacements must align with original records")
        original_ids = {record.record_id for record in originals}
        embedded = (
            self._connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE record_id IN (%s)" % ",".join("?" * len(original_ids)), tuple(original_ids)
            ).fetchone()[0]
            if original_ids
            else 0
        )
        if embedded:
            raise ValueError("Cannot replace chunks that already have vectors")
        new_records: list[SpoolRecord] = []
        created_paths: list[str] = []
        try:
            for original, documents in zip(originals, replacements):
                for output_ordinal, document in enumerate(documents):
                    text = document.page_content
                    record_id = self.stable_id(original.record_id, original.ordinal, output_ordinal, text)
                    metadata = dict(original.metadata)
                    metadata.update(dict(document.metadata))
                    metadata["chunk_id"] = record_id
                    text_path = f"rechunk/{record_id}.txt"
                    path = self.root / text_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
                    created_paths.append(text_path)
                    record = SpoolRecord(record_id, original.source, original.ordinal, text_path, metadata, len(text.encode("utf-8")))
                    new_records.append(record)
        except Exception:
            for text_path in created_paths:
                try:
                    self._file(text_path).unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
            raise
        if not originals:
            return ()
        positions = self._connection.execute(
            "SELECT position, record_id, source, ordinal, text_path, metadata, text_bytes FROM records WHERE stage='chunks' AND record_id IN (%s) ORDER BY position"
            % ",".join("?" * len(original_ids)),
            tuple(original_ids),
        ).fetchall()
        if len(positions) != len(originals) or [row[0] for row in positions] != list(range(positions[0][0], positions[0][0] + len(positions))):
            raise ValueError("Chunk replacements must target one contiguous staged batch")
        if [row[1] for row in positions] != [record.record_id for record in originals]:
            raise ValueError("Chunk replacements must preserve staged batch order")
        for row, original in zip(positions, originals):
            if (row[2], row[3], row[4], row[5], row[6]) != (
                original.source,
                original.ordinal,
                original.text_path,
                json.dumps(original.metadata, sort_keys=True),
                original.text_bytes,
            ):
                raise ValueError("Chunk replacement record does not match its staged source")
        first_position, last_position = positions[0][0], positions[-1][0]
        delta = len(new_records) - len(originals)
        with self._connection:
            # SQLite checks the primary-key uniqueness during UPDATE. Move only
            # the affected suffix out of the way, then restore the tail shifted
            # by the replacement delta; never materialize/rewrite the corpus.
            maximum = self._connection.execute("SELECT COALESCE(MAX(position), 0) FROM records WHERE stage='chunks'").fetchone()[0]
            offset = maximum + len(new_records) + 1
            self._connection.execute("UPDATE records SET position=position+? WHERE stage='chunks' AND position>=?", (offset, first_position))
            self._connection.execute("DELETE FROM records WHERE stage='chunks' AND position BETWEEN ? AND ?", (first_position + offset, last_position + offset))
            self._connection.execute("UPDATE records SET position=position-?+? WHERE stage='chunks' AND position>?", (offset, delta, last_position + offset))
            for position, record in enumerate(new_records, first_position):
                self._connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "chunks",
                        position,
                        record.record_id,
                        record.source,
                        record.ordinal,
                        record.text_path,
                        json.dumps(record.metadata, sort_keys=True),
                        record.text_bytes,
                    ),
                )
        # Rechunked sources are no longer needed once no record references them.
        # Check each path after the atomic journal update so shared task files
        # cannot be reclaimed while another record still needs them.
        for original in originals:
            remaining = self._connection.execute("SELECT 1 FROM records WHERE text_path=? LIMIT 1", (original.text_path,)).fetchone()
            if remaining is None:
                try:
                    self._file(original.text_path).unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
        return tuple(new_records)

    def accept_embeddings(self, batch: EmbeddedBatch) -> None:
        if batch.rows != len(batch.records) or batch.rows <= 0 or batch.dimensions <= 0:
            raise ValueError("Embedding rows, records, and dimensions must align")
        vector = self._file(batch.vectors_path)
        expected = batch.vector_offset_bytes + batch.rows * batch.dimensions * 4
        if not vector.is_file() or vector.stat().st_size < expected:
            raise ValueError("Embedding vector file is truncated")
        # A retried/resumed batch may overlap already committed records.  It is
        # safe only when all of them are already committed; mixed batches would
        # silently shift text/vector alignment.
        existing = self._connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE record_id IN (%s)" % ",".join("?" * len(batch.records)),
            tuple(record.record_id for record in batch.records),
        ).fetchone()[0]
        if existing == len(batch.records):
            return
        if existing:
            raise ValueError("Embedding batch overlaps an incomplete committed batch")
        try:
            with self._connection:
                position = self._connection.execute("SELECT COALESCE(MAX(position) + 1, 0) FROM embeddings").fetchone()[0]
                for offset, record in enumerate(batch.records):
                    self._connection.execute(
                        "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)",
                        (position + offset, record.record_id, batch.vectors_path, batch.vector_offset_bytes + offset * batch.dimensions * 4, batch.dimensions),
                    )
        except Exception:
            self._connection.rollback()
            raise

    def iter_embeddings(self, *, max_rows: int) -> Iterator[EmbeddedBatch]:
        rows = self._connection.execute(
            "SELECT e.record_id,r.source,r.ordinal,r.text_path,r.metadata,r.text_bytes,e.vectors_path,e.vector_offset,e.dimensions FROM embeddings e JOIN records r ON r.record_id=e.record_id AND r.stage='chunks' ORDER BY e.position"
        )
        group: list[SpoolRecord] = []
        path: str | None = None
        offset = dimensions = 0
        for row in rows:
            record = SpoolRecord(row[0], row[1], row[2], row[3], json.loads(row[4]), row[5])
            if group and (len(group) >= max_rows or row[6] != path or row[8] != dimensions or row[7] != offset + len(group) * dimensions * 4):
                assert path is not None
                yield EmbeddedBatch(tuple(group), path, offset, len(group), dimensions)
                group = []
            if not group:
                path, offset, dimensions = row[6], row[7], row[8]
            group.append(record)
        if group:
            assert path is not None
            yield EmbeddedBatch(tuple(group), path, offset, len(group), dimensions)

    def summary(self) -> dict[str, int]:
        documents = self._connection.execute("SELECT COUNT(*) FROM records WHERE stage='documents'").fetchone()[0]
        chunks, text = self._connection.execute("SELECT COUNT(*),COALESCE(SUM(text_bytes),0) FROM records WHERE stage='chunks'").fetchone()
        dimensions = self._connection.execute("SELECT COALESCE(MAX(dimensions),0) FROM embeddings").fetchone()[0]
        embedded = self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {
            "document_count": documents,
            "chunk_count": chunks,
            "embedded_count": embedded,
            "dimensions": dimensions,
            "text_bytes": text,
            "vector_bytes": embedded * dimensions * 4,
        }

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:  # journals created before state values were JSON
            return row[0]

    def set_state(self, key: str, value: Any) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value, sort_keys=True))
            )

    def is_stage_complete(self, stage: str) -> bool:
        return bool(self.get_state(f"stage:{stage}:complete", False))

    def mark_stage_complete(self, stage: str) -> None:
        self.set_state(f"stage:{stage}:complete", True)

    def _reconcile_task_files(self) -> None:
        tasks = self.root / "tasks"
        if not tasks.is_dir() or tasks.is_symlink():
            return
        referenced = {row[0] for row in self._connection.execute("SELECT text_path FROM records")}
        for task in tasks.iterdir():
            if task.is_symlink() or not task.is_dir():
                continue
            prefix = str(task.relative_to(self.root)) + "/"
            if not any(path.startswith(prefix) for path in referenced):
                shutil.rmtree(task)

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()
