import type React from 'react';
import {
  ChevronDown,
  ChevronRight,
  MoreHorizontal,
  Circle,
  CircleCheck,
  CircleAlert,
} from 'lucide-react';

export interface ToolGroupSectionProps {
  sectionId: string;
  title: string;
  description?: string;
  toolCount: number;
  /** Tools in the section before any filter; shown as "N of M tools" when it differs. */
  totalCount?: number;
  disabledCount: number;
  health: { tone: 'error' | 'success' | 'neutral'; label: string } | null;
  collapsed: boolean;
  onToggle: (sectionId: string) => void;
  onOpenActions?: (sectionId: string, anchor: HTMLElement) => void;
  actionsMenuOpen?: boolean;
  onHeaderContextMenu?: (e: React.MouseEvent) => void;
  dropTarget?: 'header' | 'body' | null;
  dropLabel?: string;
  headerDropHandlers?: Pick<
    React.HTMLAttributes<HTMLElement>,
    'onDragOver' | 'onDragLeave' | 'onDrop'
  >;
  bodyDropHandlers?: Pick<
    React.HTMLAttributes<HTMLElement>,
    'onDragOver' | 'onDragLeave' | 'onDrop'
  >;
  emptyMessage?: React.ReactNode;
  children?: React.ReactNode;
}

export function ToolGroupSection({
  sectionId,
  title,
  description,
  toolCount,
  totalCount,
  disabledCount,
  health,
  collapsed,
  onToggle,
  onOpenActions,
  actionsMenuOpen,
  onHeaderContextMenu,
  dropTarget,
  dropLabel,
  headerDropHandlers,
  bodyDropHandlers,
  emptyMessage,
  children,
}: ToolGroupSectionProps) {
  const id = `tools-section-${sectionId}`;
  const HealthIcon =
    health?.tone === 'error' ? CircleAlert : health?.tone === 'success' ? CircleCheck : Circle;
  return (
    <section
      id={id}
      data-tool-section={sectionId}
      aria-labelledby={`${id}-toggle`}
      className="tools-section"
    >
      <div
        className={`tools-section-header${dropTarget === 'header' ? ' is-drop-target' : ''}`}
        onContextMenu={onHeaderContextMenu}
        {...headerDropHandlers}
      >
        <h3 className="tools-section-heading">
          <button
            type="button"
            id={`${id}-toggle`}
            aria-expanded={!collapsed}
            aria-controls={`${id}-body`}
            aria-describedby={description && !collapsed ? `${id}-description` : undefined}
            onClick={() => onToggle(sectionId)}
          >
            {collapsed ? <ChevronRight aria-hidden="true" /> : <ChevronDown aria-hidden="true" />}
            <span className="tools-section-title" title={title}>
              {title}
            </span>
            <span className="tools-section-count">
              {totalCount !== undefined && totalCount !== toolCount
                ? `${toolCount} of ${totalCount} tools`
                : `${toolCount} ${toolCount === 1 ? 'tool' : 'tools'}`}
              {disabledCount > 0 ? `, ${disabledCount} disabled` : ''}
            </span>
          </button>
        </h3>
        {health && (
          <span className="tools-section-health" data-tone={health.tone}>
            <HealthIcon aria-hidden="true" />
            {health.label}
          </span>
        )}
        {dropTarget === 'header' && dropLabel && (
          <span className="tools-section-drop-label">{dropLabel}</span>
        )}
        {!onOpenActions && <span className="tools-section-actions-spacer" aria-hidden="true" />}
        {onOpenActions && (
          <button
            type="button"
            className="tools-section-actions"
            aria-label={`Group actions for ${title}`}
            aria-haspopup="menu"
            aria-expanded={actionsMenuOpen ?? false}
            onClick={(event) => onOpenActions(sectionId, event.currentTarget)}
          >
            <MoreHorizontal aria-hidden="true" />
          </button>
        )}
      </div>
      {description && !collapsed && (
        <p id={`${id}-description`} className="tools-section-description">
          {description}
        </p>
      )}
      <div
        id={`${id}-body`}
        className={`tools-section-body${dropTarget === 'body' ? ' is-drop-target' : ''}`}
        hidden={collapsed}
        {...bodyDropHandlers}
      >
        {toolCount === 0 && emptyMessage ? (
          <div className="tools-section-empty">{emptyMessage}</div>
        ) : (
          children
        )}
      </div>
    </section>
  );
}
