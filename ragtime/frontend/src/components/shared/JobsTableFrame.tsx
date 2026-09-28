import type { ReactNode } from 'react';

interface JobsTableFrameProps {
  children: ReactNode;
  id?: string;
  tableClassName?: string;
  wrapperClassName?: string;
}

export function JobsTableFrame({
  children,
  id,
  tableClassName = '',
  wrapperClassName = '',
}: JobsTableFrameProps) {
  return (
    <div
      id={id ? `${id}-frame` : undefined}
      className={`jobs-table-wrapper ${wrapperClassName}`.trim()}
    >
      <table id={id} className={`jobs-table ${tableClassName}`.trim()}>
        {children}
      </table>
    </div>
  );
}
