import type { ReactNode } from 'react';
import { ArrowDown, ArrowUp } from 'lucide-react';

export type SortDirection = 'asc' | 'desc';

export interface TableSortConfig<K extends string> {
  key: K;
  direction: SortDirection;
}

export interface DataTableColumn<Row, K extends string> {
  key: K;
  label: ReactNode;
  sortable?: boolean;
  headerClassName?: string;
  cellClassName?: string;
  headerTitle?: string;
  renderCell?: (row: Row, rowIndex: number) => ReactNode;
}

interface DataTableProps<Row, K extends string> {
  rows: Row[];
  columns: DataTableColumn<Row, K>[];
  sortConfig?: TableSortConfig<K> | null;
  onSort?: (key: K) => void;
  renderRow?: (row: Row, rowIndex: number) => ReactNode;
  getRowKey?: (row: Row, rowIndex: number) => string;
  wrapperClassName?: string;
  tableClassName?: string;
  emptyState?: ReactNode;
}

export function DataTable<Row, K extends string>({
  rows,
  columns,
  sortConfig = null,
  onSort,
  renderRow,
  getRowKey,
  wrapperClassName = 'jobs-table-wrapper',
  tableClassName = 'jobs-table users-table-compact',
  emptyState = null,
}: DataTableProps<Row, K>) {
  if (rows.length === 0 && emptyState) {
    return <>{emptyState}</>;
  }

  return (
    <div className={wrapperClassName}>
      <table className={tableClassName}>
        <thead>
          <tr>
            {columns.map((col) => {
              const active = sortConfig?.key === col.key;
              const sortable = col.sortable !== false && Boolean(onSort);
              const handleSort = sortable && onSort ? () => onSort(col.key) : undefined;

              return (
                <th
                  key={col.key}
                  className={col.headerClassName}
                  title={col.headerTitle}
                  onClick={handleSort}
                  style={sortable ? { cursor: 'pointer' } : undefined}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {col.label}
                    {active && sortConfig && (sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                  </div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {renderRow
            ? rows.map((row, idx) => renderRow(row, idx))
            : rows.map((row, idx) => (
              <tr key={getRowKey ? getRowKey(row, idx) : String(idx)}>
                {columns.map((col) => (
                  <td key={col.key} className={col.cellClassName}>
                    {col.renderCell ? col.renderCell(row, idx) : null}
                  </td>
                ))}
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}
