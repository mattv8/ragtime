from enum import Enum
from typing import Any

class AuthProvider(str, Enum):
    ldap = "ldap"
    local = "local"
    local_managed = "local_managed"

class ChatTaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    interrupted = "interrupted"

class FilesystemIndexStatus(str, Enum):
    pending = "pending"
    indexing = "indexing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

class IndexStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class ToolType(str, Enum):
    postgres = "postgres"
    mysql = "mysql"
    mssql = "mssql"
    influxdb = "influxdb"
    odoo_shell = "odoo_shell"
    ssh_shell = "ssh_shell"
    filesystem_indexer = "filesystem_indexer"
    solidworks_pdm = "solidworks_pdm"

class UsageAttemptStatus(str, Enum):
    started = "started"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    interrupted = "interrupted"

class UserRole(str, Enum):
    user = "user"
    admin = "admin"

class VectorStoreType(str, Enum):
    pgvector = "pgvector"
    faiss = "faiss"

class WorkspaceRole(str, Enum):
    owner = "owner"
    editor = "editor"
    viewer = "viewer"

def __getattr__(name: str) -> Any: ...
