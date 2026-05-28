import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  ChevronRight,
  Database,
  Download,
  FileText,
  Filter,
  Loader2,
  PlusCircle,
  RefreshCw,
  Save,
  Sparkles,
  Table as TableIcon,
  Terminal,
  Trash2,
  Upload,
  X,
} from 'lucide-react';

import { api } from '@/api/client';
import type {
  SqliteInspectorAlterationStep,
  SqliteInspectorColumnInfo,
  SqliteInspectorColumnSpec,
  SqliteInspectorColumnType,
  SqliteInspectorDatabaseSummary,
  SqliteInspectorRowPage,
  SqliteInspectorSqlQueryResponse,
  SqliteInspectorTableSchema,
  SqliteInspectorTableSummary,
} from '@/types';

import { DeleteConfirmButton } from '../DeleteConfirmButton';
import { ToastContainer, useToast } from './Toast';

const COLUMN_TYPES: SqliteInspectorColumnType[] = ['TEXT', 'INTEGER', 'REAL', 'NUMERIC', 'BLOB'];
const DEFAULT_ROW_PAGE_SIZE = 50;

type WizardStep = 'databases' | 'database' | 'table';
type TableSubTab = 'rows' | 'schema' | 'console';

interface WorkspaceSqliteInspectorModalProps {
  isOpen: boolean;
  workspaceId: string | null;
  workspaceName?: string;
  canEdit: boolean;
  onClose: () => void;
  onPersistencePromoted?: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatTimestamp(ms: number): string {
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return '';
  }
}

function renderCellValue(value: unknown): string {
  if (value === null || value === undefined) return 'NULL';
  if (typeof value === 'object') {
    if ((value as { __blob__?: boolean }).__blob__) {
      const size = (value as { size?: number }).size ?? 0;
      return `<blob ${size}B>`;
    }
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  if (typeof value === 'string') {
    return value;
  }
  return String(value);
}

function dbDisplayName(name: string): string {
  return name
    .replace(/\.(sqlite3?|db3?)$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function parseCellInput(raw: string, type: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === '' || trimmed.toUpperCase() === 'NULL') return null;
  const upper = (type || '').toUpperCase();
  if (upper.includes('INT')) {
    const n = Number(trimmed);
    return Number.isFinite(n) ? Math.trunc(n) : raw;
  }
  if (upper.includes('REAL') || upper.includes('FLOA') || upper.includes('DOUB') || upper.includes('NUMERIC')) {
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : raw;
  }
  return raw;
}

type FilterOperator = 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'gt' | 'gte' | 'lt' | 'lte' | 'is_null' | 'is_not_null';

interface FilterCondition {
  id: string;
  column: string;
  operator: FilterOperator;
  value: string;
}

const FILTER_OPERATORS: { value: FilterOperator; label: string; needsValue: boolean }[] = [
  { value: 'equals', label: 'equals', needsValue: true },
  { value: 'not_equals', label: 'not equals', needsValue: true },
  { value: 'contains', label: 'contains', needsValue: true },
  { value: 'not_contains', label: 'not contains', needsValue: true },
  { value: 'gt', label: '>', needsValue: true },
  { value: 'gte', label: '>=', needsValue: true },
  { value: 'lt', label: '<', needsValue: true },
  { value: 'lte', label: '<=', needsValue: true },
  { value: 'is_null', label: 'is null', needsValue: false },
  { value: 'is_not_null', label: 'is not null', needsValue: false },
];

function applyFilterCondition(row: Record<string, unknown>, condition: FilterCondition): boolean {
  if (!condition.column) return true;
  const value = row[condition.column];
  const strVal = renderCellValue(value).toLowerCase();
  const fv = condition.value.toLowerCase();
  switch (condition.operator) {
    case 'equals': return strVal === fv;
    case 'not_equals': return strVal !== fv;
    case 'contains': return strVal.includes(fv);
    case 'not_contains': return !strVal.includes(fv);
    case 'gt': { const n = Number(value); const f = Number(condition.value); return Number.isFinite(n) && Number.isFinite(f) ? n > f : String(value) > condition.value; }
    case 'gte': { const n = Number(value); const f = Number(condition.value); return Number.isFinite(n) && Number.isFinite(f) ? n >= f : String(value) >= condition.value; }
    case 'lt': { const n = Number(value); const f = Number(condition.value); return Number.isFinite(n) && Number.isFinite(f) ? n < f : String(value) < condition.value; }
    case 'lte': { const n = Number(value); const f = Number(condition.value); return Number.isFinite(n) && Number.isFinite(f) ? n <= f : String(value) <= condition.value; }
    case 'is_null': return value === null || value === undefined;
    case 'is_not_null': return value !== null && value !== undefined;
    default: return true;
  }
}

function buildSqlFromFilters(tableName: string, filters: FilterCondition[]): string {
  const table = `"${tableName.replace(/"/g, '""')}"`;
  const active = filters.filter(
    (f) => f.column && (FILTER_OPERATORS.find((o) => o.value === f.operator)?.needsValue === false || f.value.trim()),
  );
  if (active.length === 0) return `SELECT * FROM ${table} LIMIT 50;`;
  const clauses = active.map((f) => {
    const col = `"${f.column.replace(/"/g, '""')}"`;
    const v = f.value.replace(/'/g, "''");
    switch (f.operator) {
      case 'equals': return `${col} = '${v}'`;
      case 'not_equals': return `${col} != '${v}'`;
      case 'contains': return `${col} LIKE '%${v}%'`;
      case 'not_contains': return `${col} NOT LIKE '%${v}%'`;
      case 'gt': return `${col} > ${f.value}`;
      case 'gte': return `${col} >= ${f.value}`;
      case 'lt': return `${col} < ${f.value}`;
      case 'lte': return `${col} <= ${f.value}`;
      case 'is_null': return `${col} IS NULL`;
      case 'is_not_null': return `${col} IS NOT NULL`;
      default: return '';
    }
  }).filter(Boolean);
  return `SELECT * FROM ${table}\nWHERE ${clauses.join('\n  AND ')}\nLIMIT 50;`;
}

export function WorkspaceSqliteInspectorModal({
  isOpen,
  workspaceId,
  workspaceName,
  canEdit,
  onClose,
  onPersistencePromoted,
}: WorkspaceSqliteInspectorModalProps) {
  const [toasts, toastActions] = useToast();
  const toastSuccess = toastActions.success;
  const toastError = toastActions.error;
  const toastDismiss = toastActions.dismiss;
  const databaseImportInputRef = useRef<HTMLInputElement | null>(null);
  const tableImportInputRef = useRef<HTMLInputElement | null>(null);
  const [pendingDatabaseImportName, setPendingDatabaseImportName] = useState<string | null>(null);
  const [pendingTableImportName, setPendingTableImportName] = useState<string | null>(null);
  const [step, setStep] = useState<WizardStep>('databases');
  const [databases, setDatabases] = useState<SqliteInspectorDatabaseSummary[]>([]);
  const [defaultDatabaseName, setDefaultDatabaseName] = useState<string>('app.sqlite3');
  const [persistenceMode, setPersistenceMode] = useState<'include' | 'exclude'>('include');
  const [totalBytes, setTotalBytes] = useState<number>(0);
  const [loadingDatabases, setLoadingDatabases] = useState(false);

  const [selectedDatabase, setSelectedDatabase] = useState<SqliteInspectorDatabaseSummary | null>(null);
  const [tables, setTables] = useState<SqliteInspectorTableSummary[]>([]);
  const [loadingTables, setLoadingTables] = useState(false);

  const [selectedTableName, setSelectedTableName] = useState<string | null>(null);
  const [tableSubTab, setTableSubTab] = useState<TableSubTab>('rows');
  const [rowPage, setRowPage] = useState<SqliteInspectorRowPage | null>(null);
  const [loadingRows, setLoadingRows] = useState(false);
  const [rowOffset, setRowOffset] = useState(0);
  const [rowLimit, setRowLimit] = useState(DEFAULT_ROW_PAGE_SIZE);
  const [tableSchema, setTableSchema] = useState<SqliteInspectorTableSchema | null>(null);
  const [loadingSchema, setLoadingSchema] = useState(false);
  const [sqlText, setSqlText] = useState('');
  const [sqlResult, setSqlResult] = useState<SqliteInspectorSqlQueryResponse | null>(null);
  const [sqlRunning, setSqlRunning] = useState(false);

  // Create-table form state
  const [showCreateTableForm, setShowCreateTableForm] = useState(false);
  const [newTableName, setNewTableName] = useState('');
  const [newTableColumns, setNewTableColumns] = useState<SqliteInspectorColumnSpec[]>([
    { name: 'id', type: 'INTEGER', primary_key: true, not_null: true },
  ]);
  const [creatingTable, setCreatingTable] = useState(false);

  // Add row form state
  const [showAddRowForm, setShowAddRowForm] = useState(false);
  const [newRowValues, setNewRowValues] = useState<Record<string, string>>({});
  const [savingRow, setSavingRow] = useState(false);


  const reset = useCallback(() => {
    setStep('databases');
    setDatabases([]);
    setSelectedDatabase(null);
    setTables([]);
    setSelectedTableName(null);
    setTableSubTab('rows');
    setRowPage(null);
    setRowOffset(0);
    setRowLimit(DEFAULT_ROW_PAGE_SIZE);
    setTableSchema(null);
    setSqlText('');
    setSqlResult(null);
    setSqlRunning(false);
    setPendingDatabaseImportName(null);
    setPendingTableImportName(null);
    setShowCreateTableForm(false);
    setNewTableName('');
    setNewTableColumns([{ name: 'id', type: 'INTEGER', primary_key: true, not_null: true }]);
    setShowAddRowForm(false);
    setNewRowValues({});
  }, []);

  const loadDatabases = useCallback(async () => {
    if (!workspaceId) return;
    setLoadingDatabases(true);
    try {
      const result = await api.listUserSpaceSqliteDatabases(workspaceId);
      setDatabases(result.databases);
      setDefaultDatabaseName(result.default_database_name);
      setPersistenceMode(result.persistence_mode);
      setTotalBytes(result.total_bytes);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to load databases');
    } finally {
      setLoadingDatabases(false);
    }
  }, [toastError, workspaceId]);

  const loadTables = useCallback(async (database: SqliteInspectorDatabaseSummary) => {
    if (!workspaceId) return;
    setLoadingTables(true);
    try {
      const result = await api.listUserSpaceSqliteTables(workspaceId, database.name);
      setSelectedDatabase(result.database);
      setTables(result.tables);
      setPersistenceMode(result.persistence_mode);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to load tables');
    } finally {
      setLoadingTables(false);
    }
  }, [toastError, workspaceId]);

  const loadRows = useCallback(async (databaseName: string, tableName: string, offset: number, limit: number) => {
    if (!workspaceId) return;
    setLoadingRows(true);
    try {
      const result = await api.listUserSpaceSqliteRows(workspaceId, databaseName, tableName, { offset, limit });
      setRowPage(result);
      setRowOffset(result.offset);
      setRowLimit(result.limit);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to load rows');
    } finally {
      setLoadingRows(false);
    }
  }, [toastError, workspaceId]);

  const loadSchema = useCallback(async (databaseName: string, tableName: string) => {
    if (!workspaceId) return;
    setLoadingSchema(true);
    try {
      const result = await api.getUserSpaceSqliteTableSchema(workspaceId, databaseName, tableName);
      setTableSchema(result.schema);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to load schema');
    } finally {
      setLoadingSchema(false);
    }
  }, [toastError, workspaceId]);

  useEffect(() => {
    if (!isOpen || !workspaceId) return;
    const events = new EventSource(`/indexes/userspace/workspaces/${encodeURIComponent(workspaceId)}/sqlite/databases/events`);
    events.addEventListener('databases', (event) => {
      try {
        const result = JSON.parse((event as MessageEvent).data) as {
          databases: SqliteInspectorDatabaseSummary[];
          default_database_name: string;
          persistence_mode: 'include' | 'exclude';
          total_bytes: number;
        };
        setDatabases(result.databases);
        setDefaultDatabaseName(result.default_database_name);
        setPersistenceMode(result.persistence_mode);
        setTotalBytes(result.total_bytes);
        setSelectedDatabase((current) => {
          if (!current) return current;
          return result.databases.find((db) => db.name === current.name) ?? current;
        });
      } catch {
        // Ignore malformed event payloads; the normal request path still reports errors.
      }
    });
    return () => events.close();
  }, [isOpen, workspaceId]);

  useEffect(() => {
    if (!isOpen) {
      reset();
      return;
    }
    void loadDatabases();
  }, [isOpen, loadDatabases, reset]);

  const handlePromotionFlag = useCallback((promoted: boolean) => {
    if (!promoted) return;
    setPersistenceMode('include');
    onPersistencePromoted?.();
    toastSuccess(
      'Two-lane persistence enabled: SQLite changes will be included in workspace snapshots.',
    );
  }, [onPersistencePromoted, toastSuccess]);

  const handleInitializeDatabase = useCallback(async () => {
    if (!workspaceId || !canEdit) return;
    try {
      const result = await api.initializeUserSpaceSqliteDatabase(workspaceId, {});
      // Pre-fetch the destination data before transitioning so the new step
      // renders fully populated.
      await Promise.all([
        loadDatabases(),
        loadTables(result.database),
      ]);
      handlePromotionFlag(result.mode_promoted);
      toastSuccess(`Initialized ${result.database.name}`);
      setStep('database');
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to initialize database');
    }
  }, [canEdit, handlePromotionFlag, loadDatabases, loadTables, toastSuccess, toastError, workspaceId]);

  const handleOpenDatabase = useCallback(async (database: SqliteInspectorDatabaseSummary) => {
    await loadTables(database);
    setStep('database');
  }, [loadTables]);

  const handleOpenTable = useCallback(async (table: SqliteInspectorTableSummary) => {
    if (!selectedDatabase) return;
    // Fetch rows + schema before navigating so the table view never flashes empty.
    await Promise.all([
      loadRows(selectedDatabase.name, table.name, 0, DEFAULT_ROW_PAGE_SIZE),
      loadSchema(selectedDatabase.name, table.name),
    ]);
    setSelectedTableName(table.name);
    setTableSubTab('rows');
    setSqlText(`SELECT * FROM "${table.name.replace(/"/g, '""')}" LIMIT 50;`);
    setSqlResult(null);
    setRowOffset(0);
    setStep('table');
  }, [loadRows, loadSchema, selectedDatabase]);

  const handleDeleteDatabase = useCallback(async (database: SqliteInspectorDatabaseSummary) => {
    if (!workspaceId || !canEdit) return;
    try {
      await api.deleteUserSpaceSqliteDatabase(workspaceId, database.name);
      toastSuccess(`Deleted ${database.name}`);
      await loadDatabases();
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to delete database');
    }
  }, [canEdit, loadDatabases, toastSuccess, toastError, workspaceId]);

  const handleExportDatabase = useCallback(async (database: SqliteInspectorDatabaseSummary) => {
    if (!workspaceId) return;
    try {
      await api.exportUserSpaceSqliteDatabase(workspaceId, database.name);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to export database');
    }
  }, [toastError, workspaceId]);

  const handleRequestImportDatabase = useCallback((databaseName: string) => {
    if (!canEdit) return;
    setPendingDatabaseImportName(databaseName);
    databaseImportInputRef.current?.click();
  }, [canEdit]);

  const handleDatabaseImportFile = useCallback(async (file: File | null) => {
    if (!workspaceId || !canEdit || !pendingDatabaseImportName || !file) return;
    const formData = new FormData();
    formData.append('database_file', file);
    try {
      const result = await api.importUserSpaceSqliteDatabase(workspaceId, pendingDatabaseImportName, formData);
      handlePromotionFlag(result.mode_promoted);
      toastSuccess(`Imported ${result.database.name}`);
      await loadDatabases();
      await loadTables(result.database);
      setStep('database');
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to import database');
    } finally {
      setPendingDatabaseImportName(null);
      if (databaseImportInputRef.current) databaseImportInputRef.current.value = '';
    }
  }, [canEdit, handlePromotionFlag, loadDatabases, loadTables, pendingDatabaseImportName, toastError, toastSuccess, workspaceId]);

  const handleCreateTable = useCallback(async () => {
    if (!workspaceId || !canEdit || !selectedDatabase) return;
    setCreatingTable(true);
    try {
      const cleanColumns: SqliteInspectorColumnSpec[] = newTableColumns
        .map((c) => ({
          name: c.name.trim(),
          type: c.type,
          not_null: !!c.not_null,
          primary_key: !!c.primary_key,
          default_value: c.default_value?.trim() || undefined,
        }))
        .filter((c) => c.name);
      const result = await api.createUserSpaceSqliteTable(workspaceId, selectedDatabase.name, {
        name: newTableName.trim(),
        columns: cleanColumns,
      });
      handlePromotionFlag(result.mode_promoted);
      toastSuccess(`Created table ${result.table.name}`);
      setShowCreateTableForm(false);
      setNewTableName('');
      setNewTableColumns([{ name: 'id', type: 'INTEGER', primary_key: true, not_null: true }]);
      await loadTables(selectedDatabase);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to create table');
    } finally {
      setCreatingTable(false);
    }
  }, [canEdit, handlePromotionFlag, loadTables, newTableColumns, newTableName, selectedDatabase, toastSuccess, toastError, workspaceId]);

  const handleDropTable = useCallback(async (table: SqliteInspectorTableSummary) => {
    if (!workspaceId || !canEdit || !selectedDatabase) return;
    try {
      const result = await api.dropUserSpaceSqliteTable(workspaceId, selectedDatabase.name, table.name);
      handlePromotionFlag(result.mode_promoted);
      toastSuccess(`Dropped table ${table.name}`);
      await loadTables(selectedDatabase);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to drop table');
    }
  }, [canEdit, handlePromotionFlag, loadTables, selectedDatabase, toastSuccess, toastError, workspaceId]);

  const handleExportTable = useCallback(async (table: SqliteInspectorTableSummary) => {
    if (!workspaceId || !selectedDatabase) return;
    try {
      await api.exportUserSpaceSqliteTable(workspaceId, selectedDatabase.name, table.name);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to export table');
    }
  }, [selectedDatabase, toastError, workspaceId]);

  const handleRequestImportTable = useCallback((tableName: string) => {
    if (!canEdit) return;
    setPendingTableImportName(tableName);
    tableImportInputRef.current?.click();
  }, [canEdit]);

  const handleTableImportFile = useCallback(async (file: File | null) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !pendingTableImportName || !file) return;
    const formData = new FormData();
    formData.append('csv_file', file);
    try {
      const result = await api.importUserSpaceSqliteTable(workspaceId, selectedDatabase.name, pendingTableImportName, formData);
      handlePromotionFlag(result.mode_promoted);
      toastSuccess(`Imported rows into ${result.table.name}`);
      await loadTables(selectedDatabase);
      if (selectedTableName === pendingTableImportName) {
        await loadRows(selectedDatabase.name, pendingTableImportName, rowOffset, rowLimit);
        await loadSchema(selectedDatabase.name, pendingTableImportName);
      }
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to import table');
    } finally {
      setPendingTableImportName(null);
      if (tableImportInputRef.current) tableImportInputRef.current.value = '';
    }
  }, [canEdit, handlePromotionFlag, loadRows, loadSchema, loadTables, pendingTableImportName, rowLimit, rowOffset, selectedDatabase, selectedTableName, toastError, toastSuccess, workspaceId]);

  const handleSaveNewRow = useCallback(async () => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName || !tableSchema) return;
    setSavingRow(true);
    try {
      const values: Record<string, unknown> = {};
      for (const col of tableSchema.columns) {
        const raw = newRowValues[col.name];
        if (raw === undefined) continue;
        if (raw === '' && col.primary_key && col.type.toUpperCase().includes('INT')) {
          continue; // Allow auto-increment
        }
        values[col.name] = parseCellInput(raw, col.type);
      }
      const result = await api.insertUserSpaceSqliteRow(workspaceId, selectedDatabase.name, selectedTableName, { values });
      handlePromotionFlag(result.mode_promoted);
      toastSuccess('Row inserted');
      setShowAddRowForm(false);
      setNewRowValues({});
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
      await loadTables(selectedDatabase);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to insert row');
    } finally {
      setSavingRow(false);
    }
  }, [canEdit, handlePromotionFlag, loadRows, loadTables, newRowValues, rowLimit, rowOffset, selectedDatabase, selectedTableName, tableSchema, toastSuccess, toastError, workspaceId]);

  const buildRowKey = useCallback((row: Record<string, unknown>, columns: SqliteInspectorColumnInfo[]): Record<string, unknown> => {
    const key: Record<string, unknown> = {};
    const pks = columns.filter((c) => c.primary_key);
    if (pks.length > 0) {
      for (const pk of pks) {
        key[pk.name] = row[pk.name];
      }
    } else if (row._rowid !== undefined) {
      key._rowid = row._rowid;
    }
    return key;
  }, []);

  const handleUpdateCell = useCallback(async (originalRow: Record<string, unknown>, colName: string, newValue: unknown) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName || !tableSchema) return;
    try {
      const rowKey = buildRowKey(originalRow, tableSchema.columns);
      const result = await api.updateUserSpaceSqliteRow(workspaceId, selectedDatabase.name, selectedTableName, {
        row_key: rowKey,
        values: { [colName]: newValue },
      });
      handlePromotionFlag(result.mode_promoted);
      toastSuccess('Cell updated');
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to update cell');
    }
  }, [buildRowKey, canEdit, handlePromotionFlag, loadRows, rowLimit, rowOffset, selectedDatabase, selectedTableName, tableSchema, toastSuccess, toastError, workspaceId]);

  const handleDeleteRow = useCallback(async (row: Record<string, unknown>) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName || !tableSchema) return;
    try {
      const rowKey = buildRowKey(row, tableSchema.columns);
      const result = await api.deleteUserSpaceSqliteRow(workspaceId, selectedDatabase.name, selectedTableName, { row_key: rowKey });
      handlePromotionFlag(result.mode_promoted);
      toastSuccess('Row deleted');
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
      await loadTables(selectedDatabase);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to delete row');
    }
  }, [buildRowKey, canEdit, handlePromotionFlag, loadRows, loadTables, rowLimit, rowOffset, selectedDatabase, selectedTableName, tableSchema, toastSuccess, toastError, workspaceId]);

  const handleBulkDeleteRows = useCallback(async (rows: Record<string, unknown>[]) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName || !tableSchema) return;
    try {
      for (const row of rows) {
        const rowKey = buildRowKey(row, tableSchema.columns);
        await api.deleteUserSpaceSqliteRow(workspaceId, selectedDatabase.name, selectedTableName, { row_key: rowKey });
      }
      toastSuccess(`Deleted ${rows.length} row${rows.length === 1 ? '' : 's'}`);
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
      await loadTables(selectedDatabase);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to delete rows');
    }
  }, [buildRowKey, canEdit, loadRows, loadTables, rowLimit, rowOffset, selectedDatabase, selectedTableName, tableSchema, toastSuccess, toastError, workspaceId]);

  const handleDropColumn = useCallback(async (columnName: string) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName) return;
    try {
      const alterations: SqliteInspectorAlterationStep[] = [{ op: 'drop_column', column_name: columnName }];
      const result = await api.alterUserSpaceSqliteTable(workspaceId, selectedDatabase.name, selectedTableName, { alterations });
      handlePromotionFlag(result.mode_promoted);
      setTableSchema(result.schema);
      toastSuccess(`Dropped column ${columnName}`);
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to drop column');
    }
  }, [canEdit, handlePromotionFlag, loadRows, rowLimit, rowOffset, selectedDatabase, selectedTableName, toastSuccess, toastError, workspaceId]);

  const handleAddColumn = useCallback(async (spec: SqliteInspectorColumnSpec) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName) return;
    try {
      const alterations: SqliteInspectorAlterationStep[] = [{ op: 'add_column', column: spec }];
      const result = await api.alterUserSpaceSqliteTable(workspaceId, selectedDatabase.name, selectedTableName, { alterations });
      handlePromotionFlag(result.mode_promoted);
      setTableSchema(result.schema);
      toastSuccess(`Added column ${spec.name}`);
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to add column');
    }
  }, [canEdit, handlePromotionFlag, loadRows, rowLimit, rowOffset, selectedDatabase, selectedTableName, toastSuccess, toastError, workspaceId]);

  const handleRenameColumn = useCallback(async (columnName: string, newColumnName: string) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName || columnName === newColumnName) return;
    try {
      const alterations: SqliteInspectorAlterationStep[] = [{ op: 'rename_column', column_name: columnName, new_column_name: newColumnName }];
      const result = await api.alterUserSpaceSqliteTable(workspaceId, selectedDatabase.name, selectedTableName, { alterations });
      handlePromotionFlag(result.mode_promoted);
      setTableSchema(result.schema);
      toastSuccess(`Renamed column ${columnName}`);
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to rename column');
    }
  }, [canEdit, handlePromotionFlag, loadRows, rowLimit, rowOffset, selectedDatabase, selectedTableName, toastError, toastSuccess, workspaceId]);

  const handleChangeColumnType = useCallback(async (column: SqliteInspectorColumnInfo, newType: SqliteInspectorColumnType) => {
    if (!workspaceId || !canEdit || !selectedDatabase || !selectedTableName || column.type === newType) return;
    try {
      const alterations: SqliteInspectorAlterationStep[] = [{
        op: 'change_column_type',
        column_name: column.name,
        column: {
          name: column.name,
          type: newType,
          not_null: column.not_null,
          primary_key: column.primary_key,
          default_value: column.default_value,
        },
      }];
      const result = await api.alterUserSpaceSqliteTable(workspaceId, selectedDatabase.name, selectedTableName, { alterations });
      handlePromotionFlag(result.mode_promoted);
      setTableSchema(result.schema);
      toastSuccess(`Changed ${column.name} to ${newType}`);
      await loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to change column type');
    }
  }, [canEdit, handlePromotionFlag, loadRows, rowLimit, rowOffset, selectedDatabase, selectedTableName, toastError, toastSuccess, workspaceId]);

  const handleRunSql = useCallback(async () => {
    if (!workspaceId || !selectedDatabase || !sqlText.trim()) return;
    setSqlRunning(true);
    try {
      const result = await api.queryUserSpaceSqliteDatabase(workspaceId, selectedDatabase.name, { sql: sqlText, max_rows: 200 });
      setSqlResult(result);
    } catch (err) {
      toastError(err instanceof Error ? err.message : 'Failed to run query');
    } finally {
      setSqlRunning(false);
    }
  }, [selectedDatabase, sqlText, toastError, workspaceId]);

  const breadcrumb = useMemo(() => {
    const parts: { label: string; onClick?: () => void }[] = [
      { label: 'Databases', onClick: () => setStep('databases') },
    ];
    if (step !== 'databases' && selectedDatabase) {
      parts.push({ label: dbDisplayName(selectedDatabase.name), onClick: () => setStep('database') });
    }
    if (step === 'table' && selectedTableName) {
      parts.push({ label: selectedTableName });
    }
    return parts;
  }, [selectedDatabase, selectedTableName, step]);

  if (!isOpen) {
    return null;
  }

  const modalSizeClass = step === 'table'
    ? 'modal-large userspace-sqlite-modal-large'
    : step === 'database'
      ? 'userspace-sqlite-modal-medium'
      : 'userspace-sqlite-modal-compact';

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className={`modal userspace-sqlite-modal ${modalSizeClass}`}>
        <input
          ref={databaseImportInputRef}
          type="file"
          accept=".sqlite,.sqlite3,.db,.db3,application/vnd.sqlite3,application/octet-stream"
          aria-hidden="true"
          tabIndex={-1}
          style={{ display: 'none' }}
          onChange={(event) => void handleDatabaseImportFile(event.target.files?.[0] ?? null)}
        />
        <input
          ref={tableImportInputRef}
          type="file"
          accept=".csv,text/csv"
          aria-hidden="true"
          tabIndex={-1}
          style={{ display: 'none' }}
          onChange={(event) => void handleTableImportFile(event.target.files?.[0] ?? null)}
        />
        <div className="modal-header">
          <div className="userspace-sqlite-title">
            <Database size={18} />
            <h3>
              SQLite Inspector
              {workspaceName ? <span className="userspace-sqlite-workspace-name">{workspaceName}</span> : null}
            </h3>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div className="userspace-sqlite-breadcrumb-row">
          <nav className="userspace-sqlite-breadcrumb" aria-label="Inspector breadcrumb">
            {breadcrumb.map((part, idx) => (
              <span key={`${part.label}-${idx}`} className="userspace-sqlite-breadcrumb-item">
                {idx > 0 && <ChevronRight size={14} className="userspace-sqlite-breadcrumb-sep" />}
                {part.onClick && idx < breadcrumb.length - 1 ? (
                  <button type="button" className="userspace-sqlite-breadcrumb-link" onClick={part.onClick}>
                    {part.label}
                  </button>
                ) : (
                  <span className="userspace-sqlite-breadcrumb-current">{part.label}</span>
                )}
              </span>
            ))}
          </nav>
          {step !== 'databases' && (
            <button
              type="button"
              className="btn btn-secondary btn-sm btn-icon"
              onClick={() => setStep(step === 'table' ? 'database' : 'databases')}
              title="Back"
            >
              <ArrowLeft size={14} />
            </button>
          )}
        </div>

        <div className="modal-body userspace-sqlite-body">
          {persistenceMode === 'exclude' && (
            <div className="userspace-sqlite-banner">
              <Sparkles size={14} />
              <span>This workspace currently excludes SQLite from snapshots. The first edit will switch it to two-lane persistence (include).</span>
            </div>
          )}

          {step === 'databases' && (
            <DatabasesStep
              databases={databases}
              defaultDatabaseName={defaultDatabaseName}
              totalBytes={totalBytes}
              loading={loadingDatabases}
              canEdit={canEdit}
              onInitialize={handleInitializeDatabase}
              onImportDefault={() => handleRequestImportDatabase(defaultDatabaseName)}
              onOpen={handleOpenDatabase}
              onExport={handleExportDatabase}
              onImport={handleRequestImportDatabase}
              onDelete={handleDeleteDatabase}
            />
          )}

          {step === 'database' && selectedDatabase && (
            <DatabaseStep
              database={selectedDatabase}
              tables={tables}
              loading={loadingTables}
              canEdit={canEdit}
              showCreateForm={showCreateTableForm}
              onShowCreateForm={() => setShowCreateTableForm(true)}
              onCancelCreateForm={() => setShowCreateTableForm(false)}
              newTableName={newTableName}
              onNewTableNameChange={setNewTableName}
              newTableColumns={newTableColumns}
              onNewTableColumnsChange={setNewTableColumns}
              creating={creatingTable}
              onCreate={handleCreateTable}
              onRefresh={() => loadTables(selectedDatabase)}
              onOpenTable={handleOpenTable}
              onExportTable={handleExportTable}
              onImportTable={handleRequestImportTable}
              onDropTable={handleDropTable}
              sqlText={sqlText}
              onSqlTextChange={setSqlText}
              sqlResult={sqlResult}
              sqlRunning={sqlRunning}
              onRunSql={handleRunSql}
            />
          )}

          {step === 'table' && selectedDatabase && selectedTableName && (
            <TableStep
              databaseName={selectedDatabase.name}
              tableName={selectedTableName}
              subTab={tableSubTab}
              onSubTabChange={setTableSubTab}
              rowPage={rowPage}
              loadingRows={loadingRows}
              rowOffset={rowOffset}
              rowLimit={rowLimit}
              onChangeOffset={(next) => {
                setRowOffset(next);
                void loadRows(selectedDatabase.name, selectedTableName, next, rowLimit);
              }}
              schema={tableSchema}
              loadingSchema={loadingSchema}
              canEdit={canEdit}
              showAddRowForm={showAddRowForm}
              onShowAddRowForm={() => {
                setNewRowValues({});
                setShowAddRowForm(true);
              }}
              onCancelAddRowForm={() => setShowAddRowForm(false)}
              newRowValues={newRowValues}
              onNewRowValuesChange={setNewRowValues}
              savingRow={savingRow}
              onSaveNewRow={handleSaveNewRow}
              onUpdateCell={handleUpdateCell}
              onDeleteRow={handleDeleteRow}
              onBulkDeleteRows={handleBulkDeleteRows}
              onRefreshRows={() => loadRows(selectedDatabase.name, selectedTableName, rowOffset, rowLimit)}
              onAddColumn={handleAddColumn}
              onRenameColumn={handleRenameColumn}
              onChangeColumnType={handleChangeColumnType}
              onDropColumn={handleDropColumn}
              onRefreshSchema={() => loadSchema(selectedDatabase.name, selectedTableName)}
              sqlText={sqlText}
              onSqlTextChange={setSqlText}
              sqlResult={sqlResult}
              sqlRunning={sqlRunning}
              onRunSql={handleRunSql}
            />
          )}
        </div>
      </div>
      <ToastContainer toasts={toasts} onDismiss={toastDismiss} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step components
// ---------------------------------------------------------------------------

interface DatabasesStepProps {
  databases: SqliteInspectorDatabaseSummary[];
  defaultDatabaseName: string;
  totalBytes: number;
  loading: boolean;
  canEdit: boolean;
  onInitialize: () => void;
  onImportDefault: () => void;
  onOpen: (db: SqliteInspectorDatabaseSummary) => void;
  onExport: (db: SqliteInspectorDatabaseSummary) => void;
  onImport: (databaseName: string) => void;
  onDelete: (db: SqliteInspectorDatabaseSummary) => void;
}

function DatabasesStep({
  databases,
  defaultDatabaseName,
  totalBytes,
  loading,
  canEdit,
  onInitialize,
  onImportDefault,
  onOpen,
  onExport,
  onImport,
  onDelete,
}: DatabasesStepProps) {
  const hasDefault = databases.some((d) => d.name === defaultDatabaseName);
  return (
    <div className="userspace-sqlite-step">
      <div className="userspace-sqlite-step-header">
        <div>
          <h4>Databases</h4>
          <p className="field-help">
            Files under <code>.ragtime/db/</code>. Total size: {formatBytes(totalBytes)}.
          </p>
        </div>
        <div className="userspace-sqlite-step-actions">
          {!hasDefault && canEdit && (
            <button type="button" className="btn btn-primary btn-sm" onClick={onInitialize}>
              <PlusCircle size={14} /> Initialize {defaultDatabaseName}
            </button>
          )}
          {canEdit && (
            <button type="button" className="btn btn-secondary btn-sm" onClick={onImportDefault}>
              <Upload size={14} /> Import database
            </button>
          )}
        </div>
      </div>

      {loading && !databases.length ? (
        <div className="userspace-sqlite-empty"><Loader2 size={16} className="spinning" /> Loading…</div>
      ) : databases.length === 0 ? (
        <div className="userspace-sqlite-empty">
          <Database size={24} />
          <p>No databases yet.</p>
          {canEdit && (
            <button type="button" className="btn btn-primary btn-sm" onClick={onInitialize}>
              <PlusCircle size={14} /> Initialize {defaultDatabaseName}
            </button>
          )}
        </div>
      ) : (
        <div className="userspace-sqlite-card-grid">
          {databases.map((db) => (
            <div key={db.name} className="userspace-sqlite-card">
              <button type="button" className="userspace-sqlite-card-main" onClick={() => onOpen(db)}>
                <div className="userspace-sqlite-card-title">
                  <Database size={16} />
                  <span>{db.name}</span>
                </div>
                <div className="userspace-sqlite-card-meta">
                  <span>{db.table_count} table{db.table_count === 1 ? '' : 's'}</span>
                  <span>{formatBytes(db.size_bytes)}</span>
                  <span>Modified {formatTimestamp(db.last_modified_ms)}</span>
                </div>
              </button>
              <div className="userspace-sqlite-card-actions">
                <button
                  type="button"
                  className="btn btn-secondary btn-sm btn-icon"
                  onClick={() => onExport(db)}
                  title="Export database"
                >
                  <Download size={14} />
                </button>
                {canEdit && (
                  <>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm btn-icon"
                      onClick={() => onImport(db.name)}
                      title="Import database"
                    >
                      <Upload size={14} />
                    </button>
                    <DeleteConfirmButton
                      onDelete={() => onDelete(db)}
                      className="btn btn-danger btn-sm"
                      title="Delete database"
                      buttonText="Delete"
                    />
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface DatabaseStepProps {
  database: SqliteInspectorDatabaseSummary;
  tables: SqliteInspectorTableSummary[];
  loading: boolean;
  canEdit: boolean;
  showCreateForm: boolean;
  onShowCreateForm: () => void;
  onCancelCreateForm: () => void;
  newTableName: string;
  onNewTableNameChange: (value: string) => void;
  newTableColumns: SqliteInspectorColumnSpec[];
  onNewTableColumnsChange: (cols: SqliteInspectorColumnSpec[]) => void;
  creating: boolean;
  onCreate: () => void;
  onRefresh: () => void;
  onOpenTable: (table: SqliteInspectorTableSummary) => void;
  onExportTable: (table: SqliteInspectorTableSummary) => void;
  onImportTable: (tableName: string) => void;
  onDropTable: (table: SqliteInspectorTableSummary) => void;
  sqlText: string;
  onSqlTextChange: (sql: string) => void;
  sqlResult: SqliteInspectorSqlQueryResponse | null;
  sqlRunning: boolean;
  onRunSql: () => void;
}

function DatabaseStep({
  database,
  tables,
  loading,
  canEdit,
  showCreateForm,
  onShowCreateForm,
  onCancelCreateForm,
  newTableName,
  onNewTableNameChange,
  newTableColumns,
  onNewTableColumnsChange,
  creating,
  onCreate,
  onRefresh,
  onOpenTable,
  onExportTable,
  onImportTable,
  onDropTable,
  sqlText,
  onSqlTextChange,
  sqlResult,
  sqlRunning,
  onRunSql,
}: DatabaseStepProps) {
  const [subTab, setSubTab] = useState<'tables' | 'console'>('tables');
  const updateColumn = (idx: number, patch: Partial<SqliteInspectorColumnSpec>) => {
    const next = newTableColumns.map((c, i) => (i === idx ? { ...c, ...patch } : c));
    onNewTableColumnsChange(next);
  };
  const removeColumn = (idx: number) => {
    onNewTableColumnsChange(newTableColumns.filter((_, i) => i !== idx));
  };
  const addColumn = () => {
    onNewTableColumnsChange([...newTableColumns, { name: '', type: 'TEXT' }]);
  };

  return (
    <div className="userspace-sqlite-step">
      <div className="userspace-sqlite-step-header userspace-sqlite-step-header-inline">
        <div>
          <div className="userspace-sqlite-meta">
            <span>{database.table_count} table{database.table_count === 1 ? '' : 's'}</span>
            <span>{formatBytes(database.size_bytes)}</span>
            <code className="userspace-sqlite-meta-path">{database.relative_path}</code>
          </div>
        </div>
        <div className="userspace-sqlite-step-actions">
          {subTab === 'tables' && (
            <>
              <button type="button" className="btn btn-secondary btn-sm btn-icon" onClick={onRefresh} disabled={loading} title="Refresh">
                <RefreshCw size={14} className={loading ? 'spinning' : undefined} />
              </button>
              {canEdit && !showCreateForm && (
                <button type="button" className="btn btn-primary btn-sm" onClick={onShowCreateForm}>
                  <PlusCircle size={14} /> New table
                </button>
              )}
            </>
          )}
          <div className="wizard-tabs wizard-tabs--compact">
            <button
              type="button"
              className={`wizard-tab ${subTab === 'tables' ? 'active' : ''}`}
              onClick={() => setSubTab('tables')}
            >
              <TableIcon size={12} /> Tables
            </button>
            <button
              type="button"
              className={`wizard-tab ${subTab === 'console' ? 'active' : ''}`}
              onClick={() => setSubTab('console')}
            >
              <Terminal size={12} /> Console
            </button>
          </div>
        </div>
      </div>

      {showCreateForm && (
        <div className="userspace-sqlite-form">
          <div className="form-row">
            <label className="field-label">Table name</label>
            <input
              type="text"
              className="form-control"
              value={newTableName}
              onChange={(e) => onNewTableNameChange(e.target.value)}
              placeholder="my_table"
              autoFocus
            />
          </div>
          <div className="userspace-sqlite-column-editor">
            <div className="userspace-sqlite-column-editor-header">
              <span>Columns</span>
              <button type="button" className="btn btn-secondary btn-sm" onClick={addColumn}>
                <PlusCircle size={12} /> Add column
              </button>
            </div>
            <div className="userspace-sqlite-column-rows">
              {newTableColumns.map((col, idx) => (
                <div key={idx} className="userspace-sqlite-column-row">
                  <input
                    type="text"
                    className="form-control"
                    value={col.name}
                    onChange={(e) => updateColumn(idx, { name: e.target.value })}
                    placeholder="column name"
                  />
                  <select
                    className="form-control"
                    value={col.type}
                    onChange={(e) => updateColumn(idx, { type: e.target.value as SqliteInspectorColumnType })}
                  >
                    {COLUMN_TYPES.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                  <label className="userspace-sqlite-column-flag">
                    <input
                      type="checkbox"
                      checked={!!col.primary_key}
                      onChange={(e) => updateColumn(idx, { primary_key: e.target.checked })}
                    /> PK
                  </label>
                  <label className="userspace-sqlite-column-flag">
                    <input
                      type="checkbox"
                      checked={!!col.not_null}
                      onChange={(e) => updateColumn(idx, { not_null: e.target.checked })}
                    /> NOT NULL
                  </label>
                  <input
                    type="text"
                    className="form-control"
                    value={col.default_value ?? ''}
                    onChange={(e) => updateColumn(idx, { default_value: e.target.value })}
                    placeholder="default (optional)"
                  />
                  <button
                    type="button"
                    className="btn btn-danger btn-sm btn-icon"
                    onClick={() => removeColumn(idx)}
                    disabled={newTableColumns.length === 1}
                    title="Remove column"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          </div>
          <div className="userspace-sqlite-form-actions">
            <button type="button" className="btn btn-secondary btn-sm" onClick={onCancelCreateForm} disabled={creating}>Cancel</button>
            <button type="button" className="btn btn-primary btn-sm" onClick={onCreate} disabled={creating || !newTableName.trim()}>
              {creating ? <Loader2 size={12} className="spinning" /> : <Save size={12} />} Create table
            </button>
          </div>
        </div>
      )}

      {subTab === 'console' ? (
        <SqlConsolePane
          sqlText={sqlText}
          onSqlTextChange={onSqlTextChange}
          result={sqlResult}
          running={sqlRunning}
          onRun={onRunSql}
        />
      ) : (
        <>
          {loading && !tables.length ? (
            <div className="userspace-sqlite-empty"><Loader2 size={16} className="spinning" /> Loading…</div>
          ) : tables.length === 0 ? (
            <div className="userspace-sqlite-empty">
              <TableIcon size={24} />
              <p>This database has no tables yet.</p>
            </div>
          ) : (
            <div className="userspace-sqlite-card-grid">
              {tables.map((table) => (
                <div key={table.name} className="userspace-sqlite-card">
                  <button type="button" className="userspace-sqlite-card-main" onClick={() => onOpenTable(table)}>
                    <div className="userspace-sqlite-card-title">
                      <TableIcon size={16} />
                      <span>{table.name}</span>
                      {table.type === 'view' && <span className="userspace-sqlite-card-badge">view</span>}
                    </div>
                    <div className="userspace-sqlite-card-meta">
                      <span>{table.row_count} row{table.row_count === 1 ? '' : 's'}</span>
                    </div>
                  </button>
                  <div className="userspace-sqlite-card-actions">
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm btn-icon"
                      onClick={() => onExportTable(table)}
                      title="Export table"
                    >
                      <Download size={14} />
                    </button>
                    {canEdit && table.type === 'table' && (
                      <>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm btn-icon"
                          onClick={() => onImportTable(table.name)}
                          title="Import CSV"
                        >
                          <Upload size={14} />
                        </button>
                        <DeleteConfirmButton
                          onDelete={() => onDropTable(table)}
                          className="btn btn-danger btn-sm"
                          title="Drop table"
                          buttonText="Drop"
                        />
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

interface TableStepProps {
  databaseName: string;
  tableName: string;
  subTab: TableSubTab;
  onSubTabChange: (tab: TableSubTab) => void;
  rowPage: SqliteInspectorRowPage | null;
  loadingRows: boolean;
  rowOffset: number;
  rowLimit: number;
  onChangeOffset: (next: number) => void;
  schema: SqliteInspectorTableSchema | null;
  loadingSchema: boolean;
  canEdit: boolean;
  showAddRowForm: boolean;
  onShowAddRowForm: () => void;
  onCancelAddRowForm: () => void;
  newRowValues: Record<string, string>;
  onNewRowValuesChange: (values: Record<string, string>) => void;
  savingRow: boolean;
  onSaveNewRow: () => void;
  onUpdateCell: (row: Record<string, unknown>, colName: string, newValue: unknown) => Promise<void>;
  onDeleteRow: (row: Record<string, unknown>) => void;
  onBulkDeleteRows: (rows: Record<string, unknown>[]) => void;
  onRefreshRows: () => void;
  onAddColumn: (spec: SqliteInspectorColumnSpec) => void;
  onRenameColumn: (columnName: string, newColumnName: string) => void;
  onChangeColumnType: (column: SqliteInspectorColumnInfo, newType: SqliteInspectorColumnType) => void;
  onDropColumn: (columnName: string) => void;
  onRefreshSchema: () => void;
  sqlText: string;
  onSqlTextChange: (sql: string) => void;
  sqlResult: SqliteInspectorSqlQueryResponse | null;
  sqlRunning: boolean;
  onRunSql: () => void;
}

function TableStep(props: TableStepProps) {
  const {
    databaseName,
    tableName,
    subTab,
    onSubTabChange,
    rowPage,
    loadingRows,
    rowOffset,
    rowLimit,
    onChangeOffset,
    schema,
    loadingSchema,
    canEdit,
    showAddRowForm,
    onShowAddRowForm,
    onCancelAddRowForm,
    newRowValues,
    onNewRowValuesChange,
    savingRow,
    onSaveNewRow,
    onUpdateCell,
    onDeleteRow,
    onBulkDeleteRows,
    onRefreshRows,
    onAddColumn,
    onRenameColumn,
    onChangeColumnType,
    onDropColumn,
    onRefreshSchema,
    sqlText,
    onSqlTextChange,
    sqlResult,
    sqlRunning,
    onRunSql,
  } = props;

  return (
    <div className="userspace-sqlite-step">
      <div className="userspace-sqlite-step-header">
        <div>
          <h4>{tableName}</h4>
          <div className="userspace-sqlite-meta">
            <span>{databaseName}</span>
            <span>{rowPage?.total ?? 0} row{(rowPage?.total ?? 0) === 1 ? '' : 's'}</span>
          </div>
        </div>
        <div className="wizard-tabs wizard-tabs--compact">
          <button
            type="button"
            className={`wizard-tab ${subTab === 'rows' ? 'active' : ''}`}
            onClick={() => onSubTabChange('rows')}
          >
            <FileText size={12} /> Rows
          </button>
          <button
            type="button"
            className={`wizard-tab ${subTab === 'schema' ? 'active' : ''}`}
            onClick={() => onSubTabChange('schema')}
          >
            <TableIcon size={12} /> Schema
          </button>
          <button
            type="button"
            className={`wizard-tab ${subTab === 'console' ? 'active' : ''}`}
            onClick={() => onSubTabChange('console')}
          >
            <Terminal size={12} /> Console
          </button>
        </div>
      </div>

      {subTab === 'rows' ? (
        <RowsPane
          tableName={tableName}
          rowPage={rowPage}
          loading={loadingRows}
          rowOffset={rowOffset}
          rowLimit={rowLimit}
          onChangeOffset={onChangeOffset}
          canEdit={canEdit}
          showAddRowForm={showAddRowForm}
          onShowAddRowForm={onShowAddRowForm}
          onCancelAddRowForm={onCancelAddRowForm}
          newRowValues={newRowValues}
          onNewRowValuesChange={onNewRowValuesChange}
          savingRow={savingRow}
          onSaveNewRow={onSaveNewRow}
          onUpdateCell={onUpdateCell}
          onDeleteRow={onDeleteRow}
          onBulkDeleteRows={onBulkDeleteRows}
          onOpenInSql={(sql) => { onSqlTextChange(sql); onSubTabChange('console'); }}
          onRefresh={onRefreshRows}
          schema={schema}
        />
      ) : subTab === 'schema' ? (
        <SchemaPane
          schema={schema}
          loading={loadingSchema}
          canEdit={canEdit}
          rowCount={rowPage?.total ?? 0}
          onAddColumn={onAddColumn}
          onRenameColumn={onRenameColumn}
          onChangeColumnType={onChangeColumnType}
          onDropColumn={onDropColumn}
          onRefresh={onRefreshSchema}
        />
      ) : (
        <SqlConsolePane
          sqlText={sqlText}
          onSqlTextChange={onSqlTextChange}
          result={sqlResult}
          running={sqlRunning}
          onRun={onRunSql}
        />
      )}
    </div>
  );
}

interface RowsPaneProps {
  tableName: string;
  rowPage: SqliteInspectorRowPage | null;
  schema: SqliteInspectorTableSchema | null;
  loading: boolean;
  rowOffset: number;
  rowLimit: number;
  onChangeOffset: (next: number) => void;
  canEdit: boolean;
  showAddRowForm: boolean;
  onShowAddRowForm: () => void;
  onCancelAddRowForm: () => void;
  newRowValues: Record<string, string>;
  onNewRowValuesChange: (values: Record<string, string>) => void;
  savingRow: boolean;
  onSaveNewRow: () => void;
  onUpdateCell: (row: Record<string, unknown>, colName: string, newValue: unknown) => Promise<void>;
  onDeleteRow: (row: Record<string, unknown>) => void;
  onBulkDeleteRows: (rows: Record<string, unknown>[]) => void;
  onOpenInSql: (sql: string) => void;
  onRefresh: () => void;
}

function RowsPane({
  tableName,
  rowPage,
  schema,
  loading,
  rowOffset,
  rowLimit,
  onChangeOffset,
  canEdit,
  showAddRowForm,
  onShowAddRowForm,
  onCancelAddRowForm,
  newRowValues,
  onNewRowValuesChange,
  savingRow,
  onSaveNewRow,
  onUpdateCell,
  onDeleteRow,
  onBulkDeleteRows,
  onOpenInSql,
  onRefresh,
}: RowsPaneProps) {
  const columns = rowPage?.columns ?? [];
  const rows = rowPage?.rows ?? [];
  const total = rowPage?.total ?? 0;
  const pageEnd = Math.min(total, rowOffset + rows.length);

  // Per-cell editing state — one cell active at a time
  const [editingCell, setEditingCell] = useState<{
    rowIdx: number;
    colName: string;
    original: string;
    draft: string;
  } | null>(null);

  // Row selection state
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());

  // Filter state
  const [showFilterBar, setShowFilterBar] = useState(false);
  const [filters, setFilters] = useState<FilterCondition[]>([]);

  // Clear selection when the page changes
  useEffect(() => {
    setSelectedRows(new Set());
  }, [rowPage]);

  // Compute client-side filtered rows
  const filteredRows = useMemo(() => {
    const active = filters.filter(
      (f) => f.column && (FILTER_OPERATORS.find((o) => o.value === f.operator)?.needsValue === false || f.value.trim()),
    );
    if (active.length === 0) return rows;
    return rows.filter((row) => active.every((f) => applyFilterCondition(row, f)));
  }, [rows, filters]);

  const allSelected = filteredRows.length > 0 && filteredRows.every((_, idx) => selectedRows.has(idx));
  const someSelected = filteredRows.some((_, idx) => selectedRows.has(idx));
  const selectedCount = selectedRows.size;

  const handleToggleSelectAll = () => {
    if (allSelected) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(filteredRows.map((_, idx) => idx)));
    }
  };

  const handleToggleRow = (idx: number) => {
    const next = new Set(selectedRows);
    if (next.has(idx)) next.delete(idx);
    else next.add(idx);
    setSelectedRows(next);
  };

  const addFilter = () => {
    setFilters((prev) => [
      ...prev,
      { id: Math.random().toString(36).slice(2), column: columns[0]?.name ?? '', operator: 'equals', value: '' },
    ]);
  };

  const updateFilter = (id: string, patch: Partial<FilterCondition>) => {
    setFilters((prev) => prev.map((f) => (f.id === id ? { ...f, ...patch } : f)));
  };

  const removeFilter = (id: string) => {
    setFilters((prev) => prev.filter((f) => f.id !== id));
  };

  const handleBulkDelete = () => {
    const rowsToDelete = filteredRows.filter((_, idx) => selectedRows.has(idx));
    if (rowsToDelete.length === 0) return;
    onBulkDeleteRows(rowsToDelete);
    setSelectedRows(new Set());
  };

  return (
    <div className="userspace-sqlite-rows-pane">
      <div className="userspace-sqlite-rows-toolbar">
        <div className="userspace-sqlite-rows-toolbar-left">
          {canEdit && selectedCount > 0 && (
            <DeleteConfirmButton
              onDelete={handleBulkDelete}
              className="btn btn-danger btn-sm"
              title={`Delete ${selectedCount} selected record${selectedCount === 1 ? '' : 's'}`}
              buttonText={`Delete ${selectedCount} record${selectedCount === 1 ? '' : 's'}`}
            />
          )}
          {canEdit && schema && (
            <button type="button" className="btn btn-primary btn-sm" onClick={onShowAddRowForm} disabled={showAddRowForm}>
              <PlusCircle size={14} /> Add record
            </button>
          )}
        </div>
        <div className="userspace-sqlite-rows-toolbar-right">
          <span className="userspace-sqlite-rows-count">
            {total === 0
              ? 'No rows'
              : filters.some((f) => f.column)
                ? `${filteredRows.length} of ${total} rows`
                : `${total} row${total === 1 ? '' : 's'}`}
            {rowPage?.elapsed_ms != null && ` · ${rowPage.elapsed_ms}ms`}
          </span>
          <div className="userspace-sqlite-rows-pager">
            <button
              type="button"
              className="btn btn-secondary btn-sm btn-icon"
              onClick={() => onChangeOffset(Math.max(0, rowOffset - rowLimit))}
              disabled={loading || rowOffset === 0}
              title="Previous page"
            >‹</button>
            <span className="userspace-sqlite-rows-page-label">{rowLimit}</span>
            <span className="userspace-sqlite-rows-page-label">{rowOffset}</span>
            <button
              type="button"
              className="btn btn-secondary btn-sm btn-icon"
              onClick={() => onChangeOffset(rowOffset + rowLimit)}
              disabled={loading || pageEnd >= total}
              title="Next page"
            >›</button>
          </div>
          <button
            type="button"
            className={`btn btn-secondary btn-sm btn-icon${showFilterBar || filters.length > 0 ? ' btn-active' : ''}`}
            onClick={() => {
              if (!showFilterBar && filters.length === 0) addFilter();
              setShowFilterBar((v) => !v);
            }}
            title="Filter rows"
          >
            <Filter size={14} />
            {filters.length > 0 && <span className="userspace-sqlite-filter-badge">{filters.length}</span>}
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-sm btn-icon"
            onClick={onRefresh}
            disabled={loading}
            title="Refresh"
          >
            <RefreshCw size={14} className={loading ? 'spinning' : undefined} />
          </button>
        </div>
      </div>

      {showFilterBar && (
        <div className="userspace-sqlite-filter-bar">
          {filters.length === 0 ? (
            <span className="userspace-sqlite-filter-empty">No filters.</span>
          ) : (
            filters.map((filter, idx) => {
              const opDef = FILTER_OPERATORS.find((o) => o.value === filter.operator);
              return (
                <div key={filter.id} className="userspace-sqlite-filter-row">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={() => removeFilter(filter.id)}
                    title="Remove filter"
                  >
                    <X size={12} />
                  </button>
                  <span className="userspace-sqlite-filter-keyword">{idx === 0 ? 'where' : 'and'}</span>
                  <select
                    className="form-control userspace-sqlite-filter-select"
                    value={filter.column}
                    onChange={(e) => updateFilter(filter.id, { column: e.target.value })}
                  >
                    {columns.map((col) => (
                      <option key={col.name} value={col.name}>{col.name}</option>
                    ))}
                  </select>
                  <select
                    className="form-control userspace-sqlite-filter-select"
                    value={filter.operator}
                    onChange={(e) => updateFilter(filter.id, { operator: e.target.value as FilterOperator })}
                  >
                    {FILTER_OPERATORS.map((op) => (
                      <option key={op.value} value={op.value}>{op.label}</option>
                    ))}
                  </select>
                  {opDef?.needsValue !== false && (
                    <input
                      type="text"
                      className="form-control userspace-sqlite-filter-value"
                      value={filter.value}
                      onChange={(e) => updateFilter(filter.id, { value: e.target.value })}
                      placeholder="value"
                    />
                  )}
                </div>
              );
            })
          )}
          <div className="userspace-sqlite-filter-actions">
            <button type="button" className="btn btn-secondary btn-sm" onClick={addFilter}>
              <PlusCircle size={12} /> Add filter
            </button>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => onOpenInSql(buildSqlFromFilters(tableName, filters))}>
              <Terminal size={12} /> Open in SQL
            </button>
            {filters.length > 0 && (
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setFilters([])}>
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      {showAddRowForm && schema && (
        <div className="userspace-sqlite-form">
          <div className="userspace-sqlite-row-form-grid">
            {schema.columns.map((col) => (
              <div key={col.name} className="userspace-sqlite-row-form-field">
                <label className="field-label">
                  {col.name}
                  <span className="userspace-sqlite-col-type"> ({col.type || 'ANY'}{col.primary_key ? ', PK' : ''}{col.not_null ? ', NN' : ''})</span>
                </label>
                <input
                  type="text"
                  className="form-control"
                  value={newRowValues[col.name] ?? ''}
                  onChange={(e) => onNewRowValuesChange({ ...newRowValues, [col.name]: e.target.value })}
                  placeholder={col.primary_key && col.type.toUpperCase().includes('INT') ? 'auto' : 'value or NULL'}
                />
              </div>
            ))}
          </div>
          <div className="userspace-sqlite-form-actions">
            <button type="button" className="btn btn-secondary btn-sm" onClick={onCancelAddRowForm} disabled={savingRow}>Cancel</button>
            <button type="button" className="btn btn-primary btn-sm" onClick={onSaveNewRow} disabled={savingRow}>
              {savingRow ? <Loader2 size={12} className="spinning" /> : <Save size={12} />} Insert
            </button>
          </div>
        </div>
      )}

      <div className="userspace-sqlite-table-wrapper">
        {loading && !rowPage ? (
          <div className="userspace-sqlite-empty"><Loader2 size={16} className="spinning" /> Loading…</div>
        ) : (
          <table className="userspace-sqlite-table">
            <thead>
              <tr>
                {canEdit && (
                  <th className="userspace-sqlite-checkbox-col">
                    <input
                      type="checkbox"
                      className="tool-group-checkbox userspace-sqlite-row-checkbox"
                      checked={allSelected}
                      ref={(el) => { if (el) el.indeterminate = someSelected && !allSelected; }}
                      onChange={handleToggleSelectAll}
                      title="Select all visible rows"
                    />
                  </th>
                )}
                {columns.map((col) => (
                  <th key={col.name}>
                    {col.name}
                    {col.primary_key && <span className="userspace-sqlite-col-badge">PK</span>}
                  </th>
                ))}
                {canEdit && <th className="userspace-sqlite-actions-col" />}
              </tr>
            </thead>
            <tbody>
              {filteredRows.length === 0 ? (
                <tr>
                  <td colSpan={columns.length + (canEdit ? 2 : 0)} className="userspace-sqlite-empty-row">
                    {rows.length > 0 ? 'No rows match the current filter.' : 'No rows on this page.'}
                  </td>
                </tr>
              ) : (
                filteredRows.map((row, idx) => (
                  <tr
                    key={idx}
                    className={[
                      editingCell?.rowIdx === idx ? 'is-editing' : '',
                      selectedRows.has(idx) ? 'is-selected' : '',
                    ].filter(Boolean).join(' ') || undefined}
                  >
                    {canEdit && (
                      <td className="userspace-sqlite-checkbox-col" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          className="tool-group-checkbox userspace-sqlite-row-checkbox"
                          checked={selectedRows.has(idx)}
                          onChange={() => handleToggleRow(idx)}
                        />
                      </td>
                    )}
                    {columns.map((col) => {
                      const editable = canEdit && !col.primary_key;
                      const isThisCell = editingCell?.rowIdx === idx && editingCell?.colName === col.name;
                      const colType = schema?.columns.find((c) => c.name === col.name)?.type ?? 'TEXT';
                      return (
                        <td
                          key={col.name}
                          className={editable ? 'userspace-sqlite-editable-cell' : undefined}
                          onClick={() => {
                            if (!editable || isThisCell) return;
                            const original = row[col.name] === null || row[col.name] === undefined ? '' : renderCellValue(row[col.name]);
                            setEditingCell({ rowIdx: idx, colName: col.name, original, draft: original });
                          }}
                        >
                          {isThisCell ? (
                            <input
                              type="text"
                              className="userspace-sqlite-cell-input"
                              autoFocus
                              value={editingCell.draft}
                              onClick={(event) => event.stopPropagation()}
                              onChange={(e) => setEditingCell((prev) => prev ? { ...prev, draft: e.target.value } : null)}
                              onBlur={() => {
                                const cell = editingCell;
                                setEditingCell(null);
                                if (cell && cell.draft !== cell.original) {
                                  void onUpdateCell(row, col.name, parseCellInput(cell.draft, colType));
                                }
                              }}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter') { event.preventDefault(); event.currentTarget.blur(); }
                                if (event.key === 'Escape') { event.preventDefault(); setEditingCell(null); }
                              }}
                            />
                          ) : (
                            <span className={row[col.name] === null ? 'userspace-sqlite-cell-null' : undefined}>
                              {renderCellValue(row[col.name])}
                            </span>
                          )}
                        </td>
                      );
                    })}
                    {canEdit && (
                      <td className="userspace-sqlite-actions-col" onClick={(event) => event.stopPropagation()}>
                        <DeleteConfirmButton
                          onDelete={() => onDeleteRow(row)}
                          className="btn btn-danger btn-sm"
                          title="Delete row"
                          buttonText="Delete"
                        />
                      </td>
                    )}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

interface SchemaPaneProps {
  schema: SqliteInspectorTableSchema | null;
  loading: boolean;
  canEdit: boolean;
  rowCount: number;
  onAddColumn: (spec: SqliteInspectorColumnSpec) => void;
  onRenameColumn: (columnName: string, newColumnName: string) => void;
  onChangeColumnType: (column: SqliteInspectorColumnInfo, newType: SqliteInspectorColumnType) => void;
  onDropColumn: (columnName: string) => void;
  onRefresh: () => void;
}

function SchemaPane({ schema, loading, canEdit, rowCount, onAddColumn, onRenameColumn, onChangeColumnType, onDropColumn, onRefresh }: SchemaPaneProps) {
  const [showAddColumn, setShowAddColumn] = useState(false);
  const [draftColumn, setDraftColumn] = useState<SqliteInspectorColumnSpec>({ name: '', type: 'TEXT' });
  const [submitting, setSubmitting] = useState(false);
  const [editingColumnName, setEditingColumnName] = useState<string | null>(null);
  const [editingNameDraft, setEditingNameDraft] = useState('');
  // Click-to-edit for type column
  const [editingTypeColumn, setEditingTypeColumn] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!draftColumn.name.trim()) return;
    setSubmitting(true);
    try {
      await onAddColumn({
        ...draftColumn,
        name: draftColumn.name.trim(),
        default_value: draftColumn.default_value?.trim() || undefined,
      });
      setShowAddColumn(false);
      setDraftColumn({ name: '', type: 'TEXT' });
    } finally {
      setSubmitting(false);
    }
  };

  if (loading && !schema) {
    return <div className="userspace-sqlite-empty"><Loader2 size={16} className="spinning" /> Loading…</div>;
  }
  if (!schema) {
    return <div className="userspace-sqlite-empty"><TableIcon size={24} /><p>No schema available.</p></div>;
  }

  return (
    <div className="userspace-sqlite-schema-pane">
      <div className="userspace-sqlite-rows-toolbar">
        <div />
        <div className="userspace-sqlite-rows-actions">
          <button type="button" className="btn btn-secondary btn-sm btn-icon" onClick={onRefresh} title="Refresh">
            <RefreshCw size={14} />
          </button>
          {canEdit && !showAddColumn && (
            <button type="button" className="btn btn-primary btn-sm" onClick={() => setShowAddColumn(true)}>
              <PlusCircle size={14} /> Add column
            </button>
          )}
        </div>
      </div>

      {showAddColumn && (
        <div className="userspace-sqlite-form">
          <div className="userspace-sqlite-column-row">
            <input
              type="text"
              className="form-control"
              value={draftColumn.name}
              onChange={(e) => setDraftColumn({ ...draftColumn, name: e.target.value })}
              placeholder="column name"
              autoFocus
            />
            <select
              className="form-control"
              value={draftColumn.type}
              onChange={(e) => setDraftColumn({ ...draftColumn, type: e.target.value as SqliteInspectorColumnType })}
            >
              {COLUMN_TYPES.map((t) => (<option key={t} value={t}>{t}</option>))}
            </select>
            <label className="userspace-sqlite-column-flag">
              <input
                type="checkbox"
                checked={!!draftColumn.not_null}
                onChange={(e) => setDraftColumn({ ...draftColumn, not_null: e.target.checked })}
              /> NOT NULL
            </label>
            <input
              type="text"
              className="form-control"
              value={draftColumn.default_value ?? ''}
              onChange={(e) => setDraftColumn({ ...draftColumn, default_value: e.target.value })}
              placeholder="default (optional)"
            />
          </div>
          <div className="userspace-sqlite-form-actions">
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => setShowAddColumn(false)} disabled={submitting}>Cancel</button>
            <button type="button" className="btn btn-primary btn-sm" onClick={handleSubmit} disabled={submitting || !draftColumn.name.trim()}>
              {submitting ? <Loader2 size={12} className="spinning" /> : <Save size={12} />} Add
            </button>
          </div>
        </div>
      )}

      <div className="userspace-sqlite-table-wrapper">
        <table className="userspace-sqlite-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Flags</th>
              <th>Default</th>
              {canEdit && <th className="userspace-sqlite-actions-col" />}
            </tr>
          </thead>
          <tbody>
            {schema.columns.map((col) => (
              <tr key={col.name}>
                <td
                  className={canEdit && !col.primary_key ? 'userspace-sqlite-editable-cell' : undefined}
                  onClick={() => {
                    if (!canEdit || col.primary_key || editingColumnName === col.name) return;
                    setEditingColumnName(col.name);
                    setEditingNameDraft(col.name);
                  }}
                >
                  {editingColumnName === col.name ? (
                    <input
                      type="text"
                      className="userspace-sqlite-cell-input"
                      value={editingNameDraft}
                      autoFocus
                      onClick={(event) => event.stopPropagation()}
                      onChange={(event) => setEditingNameDraft(event.target.value)}
                      onBlur={() => {
                        const next = editingNameDraft.trim();
                        setEditingColumnName(null);
                        if (next && next !== col.name) onRenameColumn(col.name, next);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') event.currentTarget.blur();
                        if (event.key === 'Escape') setEditingColumnName(null);
                      }}
                    />
                  ) : (
                    col.name
                  )}
                </td>
                <td
                  className={canEdit && rowCount === 0 && !col.primary_key ? 'userspace-sqlite-editable-cell' : undefined}
                  onClick={() => {
                    if (!canEdit || rowCount > 0 || col.primary_key || editingTypeColumn === col.name) return;
                    setEditingTypeColumn(col.name);
                  }}
                >
                  {canEdit && editingTypeColumn === col.name ? (
                    <select
                      className="userspace-sqlite-cell-input userspace-sqlite-cell-select"
                      autoFocus
                      value={(COLUMN_TYPES.includes(col.type as SqliteInspectorColumnType) ? col.type : 'TEXT') as SqliteInspectorColumnType}
                      onClick={(event) => event.stopPropagation()}
                      onChange={(event) => {
                        setEditingTypeColumn(null);
                        onChangeColumnType(col, event.target.value as SqliteInspectorColumnType);
                      }}
                      onBlur={() => setEditingTypeColumn(null)}
                    >
                      {COLUMN_TYPES.map((t) => (<option key={t} value={t}>{t}</option>))}
                    </select>
                  ) : (
                    <span title={rowCount > 0 ? 'Empty the table before changing column type' : undefined}>
                      {col.type || '—'}
                    </span>
                  )}
                </td>
                <td>
                  {col.primary_key && <span className="userspace-sqlite-col-badge">PK</span>}
                  {col.not_null && <span className="userspace-sqlite-col-badge">NOT NULL</span>}
                </td>
                <td>{col.default_value ?? <span className="userspace-sqlite-cell-null">NULL</span>}</td>
                {canEdit && (
                  <td className="userspace-sqlite-actions-col">
                    {!col.primary_key && (
                      <DeleteConfirmButton
                        onDelete={() => onDropColumn(col.name)}
                        className="btn btn-danger btn-sm"
                        title="Drop column"
                        buttonText="Drop"
                      />
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {schema.indexes.length > 0 && (
        <div className="userspace-sqlite-subsection">
          <h5>Indexes</h5>
          <ul className="userspace-sqlite-list">
            {schema.indexes.map((idx) => (
              <li key={idx.name}>
                <code>{idx.name}</code> — {idx.unique ? 'UNIQUE ' : ''}({idx.columns.join(', ')})
              </li>
            ))}
          </ul>
        </div>
      )}

      {schema.foreign_keys.length > 0 && (
        <div className="userspace-sqlite-subsection">
          <h5>Foreign keys</h5>
          <ul className="userspace-sqlite-list">
            {schema.foreign_keys.map((fk) => (
              <li key={`${fk.id}-${fk.seq}`}>
                <code>{fk.from_column}</code> → <code>{fk.to_table}.{fk.to_column}</code>
                {fk.on_delete && fk.on_delete !== 'NO ACTION' ? ` ON DELETE ${fk.on_delete}` : ''}
              </li>
            ))}
          </ul>
        </div>
      )}

      {schema.sql && (
        <details className="userspace-sqlite-sql-details">
          <summary>Generated SQL</summary>
          <pre className="userspace-sqlite-sql"><code>{schema.sql}</code></pre>
        </details>
      )}
    </div>
  );
}

interface SqlConsolePaneProps {
  sqlText: string;
  onSqlTextChange: (sql: string) => void;
  result: SqliteInspectorSqlQueryResponse | null;
  running: boolean;
  onRun: () => void;
}

function SqlConsolePane({ sqlText, onSqlTextChange, result, running, onRun }: SqlConsolePaneProps) {
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      onRun();
    }
  };
  return (
    <div className="userspace-sqlite-console-pane">
      <div className="userspace-sqlite-console-editor-card">
        <textarea
          className="userspace-sqlite-console-textarea"
          value={sqlText}
          onChange={(event) => onSqlTextChange(event.target.value)}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          placeholder="SELECT * FROM your_table LIMIT 50;"
        />
        <div className="userspace-sqlite-console-toolbar">
          <span className="userspace-sqlite-console-hint">Read-only · {result ? `${result.row_count} row${result.row_count === 1 ? '' : 's'}${result.truncated ? ' (truncated)' : ''}` : '⌘/Ctrl+Enter to run'}</span>
          <button type="button" className="btn btn-primary btn-sm" onClick={onRun} disabled={running || !sqlText.trim()}>
            {running ? <Loader2 size={14} className="spinning" /> : <Terminal size={14} />} Run
          </button>
        </div>
      </div>
      <div className="userspace-sqlite-console-results">
        {!result ? (
          <div className="userspace-sqlite-empty userspace-sqlite-console-placeholder">
            <Terminal size={20} />
            <p>Run a read-only query to inspect the database.</p>
          </div>
        ) : result.rows.length === 0 ? (
          <div className="userspace-sqlite-empty userspace-sqlite-console-placeholder">
            <TableIcon size={20} />
            <p>No rows returned.</p>
          </div>
        ) : (
          <div className="userspace-sqlite-table-wrapper">
            <table className="userspace-sqlite-table">
              <thead>
                <tr>{result.columns.map((column) => (<th key={column}>{column}</th>))}</tr>
              </thead>
              <tbody>
                {result.rows.map((row, idx) => (
                  <tr key={idx}>
                    {result.columns.map((column) => (
                      <td key={column}>{renderCellValue(row[column])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
