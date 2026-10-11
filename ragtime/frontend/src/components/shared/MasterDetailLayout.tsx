import {
  useEffect,
  useRef,
  type ComponentPropsWithoutRef,
  type KeyboardEvent,
  type ReactNode,
} from 'react';

export interface MasterDetailItem {
  id: string;
  name: string;
  meta?: ReactNode;
}

type DataAttributes = { [attribute: `data-${string}`]: string | undefined };
type DivProps = ComponentPropsWithoutRef<'div'> & DataAttributes;
type ListProps = ComponentPropsWithoutRef<'ol'> & DataAttributes;

export interface MasterDetailLayoutProps {
  id: string;
  heading: string;
  headerAction?: ReactNode;
  search?: ReactNode;
  items: MasterDetailItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCloseDetail: () => void;
  detail?: ReactNode;
  detailLabelledBy?: string;
  emptyMessage?: ReactNode;
  noMatchMessage?: ReactNode;
  emptyHookAttribute?: string;
  noMatchHookAttribute?: string;
  busy?: boolean;
  rootProps?: DivProps;
  listProps?: DivProps;
  listBodyProps?: ListProps;
  detailProps?: DivProps;
  rowHookAttribute: string;
}

function joinClassNames(...classNames: Array<string | undefined>): string {
  return classNames.filter(Boolean).join(' ');
}

function focusElementAfterRender(element: HTMLElement | null, root: HTMLElement | null): void {
  const activeElement = document.activeElement;
  requestAnimationFrame(() => {
    if (document.activeElement !== activeElement && activeElement?.isConnected) return;
    if (root?.contains(activeElement) && activeElement?.matches('input, textarea, select')) return;
    element?.focus();
  });
}

export function MasterDetailLayout({
  id,
  heading,
  headerAction,
  search,
  items,
  selectedId,
  onSelect,
  onCloseDetail,
  detail,
  detailLabelledBy,
  emptyMessage = 'No items configured.',
  noMatchMessage,
  emptyHookAttribute,
  noMatchHookAttribute,
  busy = false,
  rootProps,
  listProps,
  listBodyProps,
  detailProps,
  rowHookAttribute,
}: MasterDetailLayoutProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const previousSelectedId = useRef<string | null>(null);
  const mounted = useRef(false);

  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true;
      previousSelectedId.current = selectedId;
      return;
    }

    if (selectedId) {
      if (previousSelectedId.current === selectedId) return;
      previousSelectedId.current = selectedId;
      const detailHeading = detailLabelledBy ? document.getElementById(detailLabelledBy) : null;
      focusElementAfterRender(detailHeading, rootRef.current);
      return;
    }

    const closedDetailId = previousSelectedId.current;
    if (!closedDetailId) return;
    previousSelectedId.current = null;

    const root = rootRef.current;
    const row = Array.from(root?.querySelectorAll<HTMLElement>(`[${rowHookAttribute}]`) ?? []).find(
      (element) => element.getAttribute(rowHookAttribute) === closedDetailId,
    );
    if (row) {
      focusElementAfterRender(row, root);
      return;
    }
    focusElementAfterRender(
      root?.querySelector<HTMLElement>('[data-new-access-level], [data-new-category]') ?? null,
      root,
    );
  }, [detailLabelledBy, rowHookAttribute, selectedId]);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    rootProps?.onKeyDown?.(event);
    if (event.key !== 'Escape' || event.defaultPrevented || busy || !selectedId) return;
    event.preventDefault();
    onCloseDetail();
  };

  const rootClassName = joinClassNames('access-md', rootProps?.className);
  const listClassName = joinClassNames('access-md-list', listProps?.className);
  const listBodyClassName = joinClassNames('access-md-list-body', listBodyProps?.className);
  const detailClassName = joinClassNames('access-md-detail', detailProps?.className);
  const hasNoMatches = !items.length && noMatchMessage !== undefined;

  return (
    <div
      {...rootProps}
      ref={rootRef}
      id={id}
      className={rootClassName}
      data-detail-open={selectedId ? '' : undefined}
      onKeyDown={handleKeyDown}
    >
      <div {...listProps} className={listClassName}>
        <div className="access-levels-rail-header" {...{ [`data-${id}-rail-header`]: '' }}>
          <div className="access-levels-rail-header-top">
            <h4 className="access-levels-rail-heading">{heading}</h4>
            {headerAction}
          </div>
          {search}
        </div>
        <ol {...listBodyProps} className={listBodyClassName} {...{ [`data-${id}-list`]: '' }}>
          {!items.length && !hasNoMatches && (
            <li
              className="access-md-list-item auth-group-empty-state"
              {...{
                [`data-${id}-empty`]: '',
                ...(emptyHookAttribute && { [emptyHookAttribute]: '' }),
              }}
            >
              {emptyMessage}
            </li>
          )}
          {hasNoMatches && (
            <li
              className="access-md-list-item auth-group-empty-state"
              {...{
                [`data-${id}-no-match`]: '',
                ...(noMatchHookAttribute && { [noMatchHookAttribute]: '' }),
              }}
            >
              {noMatchMessage || 'No matching items.'}
            </li>
          )}
          {items.map((item) => {
            const isSelected = item.id === selectedId;
            return (
              <li className="access-md-list-item" key={item.id}>
                <button
                  type="button"
                  className={`access-md-rail-row${isSelected ? ' is-selected' : ''}`}
                  {...{ [rowHookAttribute]: item.id }}
                  aria-current={isSelected ? 'true' : undefined}
                  onClick={() => onSelect(item.id)}
                >
                  <span className="access-md-rail-row-name">{item.name}</span>
                  {item.meta && <span className="access-md-rail-row-meta">{item.meta}</span>}
                </button>
              </li>
            );
          })}
        </ol>
      </div>
      {detail && (
        <div
          {...detailProps}
          className={detailClassName}
          aria-labelledby={detailLabelledBy}
          aria-busy={busy || undefined}
        >
          {detail}
        </div>
      )}
    </div>
  );
}
