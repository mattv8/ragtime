"""Read-only, version-aware SQL source for SolidWorks PDM vaults."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable, Mapping, Optional

from ragtime.core.logging import get_logger

if TYPE_CHECKING:
    from ragtime.indexer.models import SolidworksPdmConnectionConfig


logger = get_logger(__name__)
_SAFE_PDM_EXTENSION_RE = re.compile(r"^\.?[A-Za-z0-9_-]{1,32}$")


@dataclass
class PdmVariableDef:
    variable_id: int
    name: str
    variable_type: int
    is_deleted: bool
    flag_free_update_all_version: bool
    flag_free_update_latest_version: bool


@dataclass
class PdmRawValue:
    variable_id: int
    variable_name: str
    configuration_id: int
    project_id: int
    revision_no: int
    value_text: str
    value_int: Optional[int]
    value_float: Optional[float]
    value_date: Optional[str]


@dataclass
class PdmConfigurationRef:
    configuration_id: int
    name: str


@dataclass
class PdmBomChild:
    document_id: int
    filename: str
    configuration: str


@dataclass
class PdmDocumentRecord:
    document_id: int
    filename: str
    latest_revision: int
    folder_paths: list[str]
    configurations: list[PdmConfigurationRef]
    resolved_values: dict[tuple[int, int], PdmRawValue]
    bom_children: list[PdmBomChild]
    membership_fallback: bool
    has_beyond_latest_values: bool


def build_pdm_extension_filter(extensions: list[str] | None, column: str) -> str:
    """Build the validated extension predicate used by the legacy extractor."""
    values = [str(extension or "").strip() for extension in (extensions or [])]
    if not values:
        return "1=1"
    unsafe = [extension for extension in values if not _SAFE_PDM_EXTENSION_RE.fullmatch(extension)]
    if unsafe:
        raise ValueError(f"Invalid PDM file extension filter: {unsafe[0]!r}")
    return " OR ".join(f"{column} LIKE '%{extension}'" for extension in values)


def _raw_value_from_row(row: Mapping[str, Any]) -> PdmRawValue:
    value_date = row.get("ValueDate")
    if isinstance(value_date, datetime):
        value_date = value_date.isoformat()
    elif value_date is not None:
        value_date = str(value_date)
    return PdmRawValue(
        variable_id=int(row["VariableID"]),
        variable_name=str(row.get("VariableName") or ""),
        configuration_id=int(row["ConfigurationID"]),
        project_id=int(row["ProjectID"]),
        revision_no=int(row["RevisionNo"]),
        value_text="" if row.get("ValueText") is None else str(row.get("ValueText")),
        value_int=None if row.get("ValueInt") is None else int(row["ValueInt"]),
        value_float=None if row.get("ValueFloat") is None else float(row["ValueFloat"]),
        value_date=value_date,
    )


def _value_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Make equivalent reductions independent of database row order."""
    value_date = row.get("ValueDate")
    if isinstance(value_date, datetime):
        value_date = value_date.isoformat()
    return (
        int(row["ProjectID"]),
        int(row["ConfigurationID"]),
        int(row["VariableID"]),
        int(row["RevisionNo"]),
        str(row.get("ValueText") or ""),
        repr(row.get("ValueInt")),
        repr(row.get("ValueFloat")),
        repr(value_date),
        str(row.get("VariableName") or ""),
    )


def resolve_latest_values(
    rows: Iterable[Mapping[str, Any]],
    latest_revision: int,
    reserved_project_ids: set[int] | frozenset[int],
) -> tuple[dict[tuple[int, int], PdmRawValue], bool]:
    """Resolve sparse PDM values to a checked-in revision, preserving clears."""
    latest = int(latest_revision)
    eligible: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    has_beyond_latest = False
    for row in sorted(rows, key=_value_sort_key):
        revision = int(row["RevisionNo"])
        if revision > latest:
            has_beyond_latest = True
            continue
        key = (int(row["ProjectID"]), int(row["ConfigurationID"]), int(row["VariableID"]))
        current = eligible.get(key)
        if current is None or revision >= int(current["RevisionNo"]):
            eligible[key] = row

    resolved: dict[tuple[int, int], PdmRawValue] = {}
    for key in sorted({(config_id, variable_id) for _, config_id, variable_id in eligible}):
        candidates = [row for (project_id, config_id, variable_id), row in eligible.items() if (config_id, variable_id) == key]
        winner = min(
            candidates,
            key=lambda row: (
                0 if int(row["ProjectID"]) in reserved_project_ids else 1,
                int(row["ProjectID"]),
            ),
        )
        resolved[key] = _raw_value_from_row(winner)
    return resolved, has_beyond_latest


def _assemble_document_records(
    document_rows: Iterable[Mapping[str, Any]],
    folders_by_document: Mapping[int, list[str]],
    memberships_by_document: Mapping[int, list[PdmConfigurationRef]],
    values_by_document: Mapping[int, list[Mapping[str, Any]]],
    reserved_project_ids: set[int],
    fallback_names: Mapping[int, str],
    bom_by_document: Mapping[int, list[PdmBomChild]],
) -> list[PdmDocumentRecord]:
    """Purely construct deterministic records from one SQL extraction batch."""
    records: list[PdmDocumentRecord] = []
    for row in document_rows:
        document_id = int(row["DocumentID"])
        latest_revision = int(row.get("LatestRevisionNo") or 1)
        values = values_by_document.get(document_id, [])
        resolved_values, has_beyond_latest = resolve_latest_values(values, latest_revision, reserved_project_ids)
        configurations = list(memberships_by_document.get(document_id, []))
        membership_fallback = not configurations
        if membership_fallback:
            configuration_ids = sorted({key[0] for key in resolved_values})
            configurations = [PdmConfigurationRef(configuration_id, fallback_names.get(configuration_id, "")) for configuration_id in configuration_ids]
        records.append(
            PdmDocumentRecord(
                document_id=document_id,
                filename=str(row.get("Filename") or ""),
                latest_revision=latest_revision,
                folder_paths=sorted(folders_by_document.get(document_id, [])),
                configurations=sorted(configurations, key=lambda item: item.configuration_id),
                resolved_values=resolved_values,
                bom_children=sorted(
                    bom_by_document.get(document_id, []),
                    key=lambda item: (item.document_id, item.configuration),
                ),
                membership_fallback=membership_fallback,
                has_beyond_latest_values=has_beyond_latest,
            )
        )
    return records


class PdmSqlSource:
    """SolidWorks PDM source adapter using batched read-only MSSQL queries."""

    def __init__(self, config: "SolidworksPdmConnectionConfig") -> None:
        self.config = config

    def _tunnel_config_dict(self) -> dict[str, Any]:
        fields = (
            "host",
            "port",
            "ssh_tunnel_host",
            "ssh_tunnel_port",
            "ssh_tunnel_user",
            "ssh_tunnel_password",
            "ssh_tunnel_key_path",
            "ssh_tunnel_key_content",
            "ssh_tunnel_key_passphrase",
        )
        return {field: getattr(self.config, field, "") for field in fields}

    def _start_tunnel(self) -> Any:
        if not getattr(self.config, "ssh_tunnel_enabled", False):
            return None
        from ragtime.core.ssh import SSHTunnel, ssh_tunnel_config_from_dict

        tunnel_config = ssh_tunnel_config_from_dict(self._tunnel_config_dict(), default_remote_port=1433)
        tunnel = SSHTunnel(tunnel_config)
        tunnel.start()
        return tunnel

    def _connect(self, tunnel: Any = None) -> Any:
        try:
            import pymssql  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("pymssql not installed for PDM database access") from exc
        connect = getattr(pymssql, "connect", None)
        if not callable(connect):
            raise RuntimeError("pymssql.connect is not available")
        return connect(
            server="127.0.0.1" if tunnel else getattr(self.config, "host"),
            port=str(tunnel.local_port if tunnel else (getattr(self.config, "port", 1433) or 1433)),
            user=getattr(self.config, "user"),
            password=getattr(self.config, "password"),
            database=getattr(self.config, "database"),
            login_timeout=30,
            timeout=300,
        )

    async def discover_variables(self) -> list[PdmVariableDef]:
        def run() -> list[PdmVariableDef]:
            tunnel = self._start_tunnel()
            conn = None
            try:
                conn = self._connect(tunnel)
                cursor = conn.cursor(as_dict=True)
                cursor.execute(
                    "SELECT VariableID, VariableName, VariableType, IsDeleted, FlagUnique, FlagMandatory, FlagFreeUpdateAllVersion, FlagFreeUpdateLatestVersion FROM Variable ORDER BY VariableID"
                )
                return [
                    PdmVariableDef(
                        variable_id=int(row["VariableID"]),
                        name=str(row.get("VariableName") or ""),
                        variable_type=int(row.get("VariableType") or 0),
                        is_deleted=bool(row.get("IsDeleted")),
                        flag_free_update_all_version=bool(row.get("FlagFreeUpdateAllVersion")),
                        flag_free_update_latest_version=bool(row.get("FlagFreeUpdateLatestVersion")),
                    )
                    for row in (cursor.fetchall() or [])
                ]
            finally:
                if conn is not None:
                    conn.close()
                if tunnel is not None:
                    tunnel.stop()

        return await asyncio.to_thread(run)

    async def count_documents(self) -> int:
        ext_filter = build_pdm_extension_filter(getattr(self.config, "file_extensions", None), "d.Filename")
        deleted_filter = " AND d.Deleted = 0" if getattr(self.config, "exclude_deleted", False) else ""

        def run() -> int:
            tunnel = self._start_tunnel()
            try:
                return self._count_documents_sync(tunnel, ext_filter, deleted_filter)
            finally:
                if tunnel is not None:
                    tunnel.stop()

        return await asyncio.to_thread(run)

    def _count_documents_sync(self, tunnel: Any, ext_filter: str, deleted_filter: str) -> int:
        conn = None
        try:
            conn = self._connect(tunnel)
            cursor = conn.cursor(as_dict=True)
            cursor.execute(f"SELECT COUNT(DISTINCT d.DocumentID) AS count FROM Documents d WHERE ({ext_filter}){deleted_filter}")
            row = cursor.fetchone()
            return int((row or {}).get("count", 0))
        finally:
            if conn is not None:
                conn.close()

    async def extract_documents_batched(
        self,
        variable_map: dict[int, str],
        max_documents: int | None = None,
        batch_size: int = 500,
        on_batch_extracted: Callable[[int, int], None] | None = None,
    ) -> AsyncIterator[list[PdmDocumentRecord]]:
        ext_filter = build_pdm_extension_filter(getattr(self.config, "file_extensions", None), "d.Filename")
        deleted_filter = " AND d.Deleted = 0" if getattr(self.config, "exclude_deleted", False) else ""
        per_batch = int(batch_size)
        if per_batch <= 0:
            raise ValueError("batch_size must be greater than zero")

        tunnel = await asyncio.to_thread(self._start_tunnel)
        try:
            total_count = await asyncio.to_thread(self._count_documents_sync, tunnel, ext_filter, deleted_filter)
            maximum = int(max_documents) if max_documents is not None else None
            if maximum is not None:
                total_count = min(total_count, maximum)
            limit_total = maximum if maximum is not None else total_count
            offset = 0
            total_extracted = 0
            while offset < limit_total:
                limit = min(per_batch, limit_total - offset)
                batch = await asyncio.to_thread(
                    self._extract_batch,
                    tunnel,
                    ext_filter,
                    deleted_filter,
                    offset,
                    limit,
                    variable_map,
                )
                if not batch:
                    break
                total_extracted += len(batch)
                if on_batch_extracted:
                    on_batch_extracted(total_extracted, total_count)
                yield batch
                offset += limit
                await asyncio.sleep(0.01)
        finally:
            if tunnel is not None:
                await asyncio.to_thread(tunnel.stop)

    def _extract_batch(
        self,
        tunnel: Any,
        ext_filter: str,
        deleted_filter: str,
        offset: int,
        limit: int,
        variable_map: Mapping[int, str],
    ) -> list[PdmDocumentRecord]:
        conn = None
        try:
            conn = self._connect(tunnel)
            cursor = conn.cursor(as_dict=True)
            cursor.execute(
                f"SELECT d.DocumentID, d.Filename, d.LatestRevisionNo FROM Documents d "
                f"WHERE ({ext_filter}){deleted_filter} ORDER BY d.DocumentID "
                f"OFFSET {int(offset)} ROWS FETCH NEXT {int(limit)} ROWS ONLY"
            )
            documents = cursor.fetchall() or []
            if not documents:
                return []
            document_ids = [int(row["DocumentID"]) for row in documents]
            document_id_sql = ",".join(str(value) for value in document_ids)
            folders: dict[int, list[str]] = {value: [] for value in document_ids}
            if getattr(self.config, "include_folder_path", False):
                cursor.execute(
                    "SELECT dip.DocumentID, p.Path FROM DocumentsInProjects dip "
                    "JOIN Projects p ON p.ProjectID = dip.ProjectID "
                    f"WHERE dip.DocumentID IN ({document_id_sql}) "
                    "AND (dip.Deleted IS NULL OR dip.Deleted = 0) AND p.Path IS NOT NULL"
                )
                for row in cursor.fetchall() or []:
                    folders[int(row["DocumentID"])].append(str(row["Path"]))

            memberships: dict[int, list[PdmConfigurationRef]] = {value: [] for value in document_ids}
            cursor.execute(
                "SELECT drc.DocumentID, drc.RevisionNo, drc.ConfigurationID, dc.ConfigurationName "
                "FROM DocumentRevisionConfiguration drc JOIN DocumentConfiguration dc "
                "ON dc.ConfigurationID = drc.ConfigurationID "
                f"WHERE drc.DocumentID IN ({document_id_sql})"
            )
            revisions = {int(row["DocumentID"]): int(row.get("LatestRevisionNo") or 1) for row in documents}
            for row in cursor.fetchall() or []:
                document_id = int(row["DocumentID"])
                if int(row["RevisionNo"]) == revisions[document_id]:
                    memberships[document_id].append(
                        PdmConfigurationRef(
                            int(row["ConfigurationID"]),
                            str(row.get("ConfigurationName") or ""),
                        )
                    )

            values: dict[int, list[Mapping[str, Any]]] = {value: [] for value in document_ids}
            reserved_project_ids: set[int] = set()
            fallback_names: dict[int, str] = {}
            if variable_map:
                variable_id_sql = ",".join(str(int(value)) for value in variable_map)
                cursor.execute(
                    "SELECT vv.DocumentID, vv.ProjectID, vv.ConfigurationID, vv.VariableID, vv.RevisionNo, "
                    "vv.ValueText, vv.ValueInt, vv.ValueFloat, vv.ValueDate FROM VariableValue vv "
                    f"WHERE vv.DocumentID IN ({document_id_sql}) AND vv.VariableID IN ({variable_id_sql})"
                )
                for row in cursor.fetchall() or []:
                    row = dict(row)
                    row["VariableName"] = variable_map.get(int(row["VariableID"]), "")
                    values[int(row["DocumentID"])].append(row)
                project_ids = sorted({int(row["ProjectID"]) for rows in values.values() for row in rows})
                if project_ids:
                    project_id_sql = ",".join(str(value) for value in project_ids)
                    cursor.execute(f"SELECT ProjectID, Path FROM Projects WHERE ProjectID IN ({project_id_sql})")
                    reserved_project_ids = {int(row["ProjectID"]) for row in (cursor.fetchall() or []) if row.get("Path") is None}
                configuration_ids = sorted({int(row["ConfigurationID"]) for rows in values.values() for row in rows})
                if configuration_ids:
                    configuration_id_sql = ",".join(str(value) for value in configuration_ids)
                    cursor.execute(f"SELECT ConfigurationID, ConfigurationName FROM DocumentConfiguration WHERE ConfigurationID IN ({configuration_id_sql})")
                    fallback_names = {int(row["ConfigurationID"]): str(row.get("ConfigurationName") or "") for row in (cursor.fetchall() or [])}

            bom: dict[int, list[PdmBomChild]] = {value: [] for value in document_ids}
            if getattr(self.config, "include_bom", False):
                assembly_ids = [
                    int(row["DocumentID"])
                    for row in documents
                    if "." in str(row.get("Filename") or "") and str(row["Filename"]).rsplit(".", 1)[-1].upper() == "SLDASM"
                ]
                if assembly_ids:
                    assembly_id_sql = ",".join(str(value) for value in assembly_ids)
                    cursor.execute(
                        "SELECT bs.SourceDocumentID, bsr.RowDocumentID AS ChildFileID, "
                        "d2.Filename AS ChildFilename, dc.ConfigurationName AS ChildConfigName "
                        "FROM BomSheets bs INNER JOIN BomSheetRow bsr ON bs.BomDocumentID = bsr.BomDocumentID "
                        "INNER JOIN Documents d2 ON bsr.RowDocumentID = d2.DocumentID "
                        "LEFT JOIN DocumentConfiguration dc ON bsr.RowConfigurationID = dc.ConfigurationID "
                        f"WHERE bs.SourceDocumentID IN ({assembly_id_sql})"
                    )
                    for row in cursor.fetchall() or []:
                        bom[int(row["SourceDocumentID"])].append(
                            PdmBomChild(
                                document_id=int(row["ChildFileID"]),
                                filename=str(row.get("ChildFilename") or ""),
                                configuration=str(row.get("ChildConfigName") or ""),
                            )
                        )
            return _assemble_document_records(
                documents,
                folders,
                memberships,
                values,
                reserved_project_ids,
                fallback_names,
                bom,
            )
        finally:
            if conn is not None:
                conn.close()
