import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';

export interface ModalTab {
  id: string;
  label: string;
  content: ReactNode;
  panelClassName?: string;
}

interface ModalTabsProps {
  idPrefix: string;
  label: string;
  tabs: ModalTab[];
  activeTabId: string;
  onChange: (tabId: string) => void;
}

export function ModalTabs({ idPrefix, label, tabs, activeTabId, onChange }: ModalTabsProps) {
  const [visitedTabIds, setVisitedTabIds] = useState(
    () => new Set(tabs.slice(0, 1).map(({ id }) => id)),
  );
  const tabRefs = useRef(new Map<string, HTMLButtonElement>());
  const tabIdsSignature = tabs.map(({ id }) => id).join('|');
  const selectedTabId = tabs.some(({ id }) => id === activeTabId) ? activeTabId : tabs[0]?.id;

  useEffect(() => {
    const tabIds = new Set(tabIdsSignature.split('|').filter(Boolean));
    setVisitedTabIds((previous) => {
      const next = new Set([...previous].filter((id) => tabIds.has(id)));
      const firstTabId = tabIdsSignature.split('|')[0];
      if (firstTabId) next.add(firstTabId);
      if (selectedTabId) next.add(selectedTabId);
      return next.size === previous.size && [...next].every((id) => previous.has(id))
        ? previous
        : next;
    });
    if (selectedTabId && selectedTabId !== activeTabId) onChange(selectedTabId);
  }, [activeTabId, onChange, selectedTabId, tabIdsSignature]);

  const selectTab = (tabId: string) => {
    setVisitedTabIds((previous) => new Set(previous).add(tabId));
    onChange(tabId);
  };

  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, tabId: string) => {
    const index = tabs.findIndex(({ id }) => id === tabId);
    let nextIndex: number | null = null;
    if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = tabs.length - 1;
    if (nextIndex === null) return;

    event.preventDefault();
    const nextTabId = tabs[nextIndex].id;
    selectTab(nextTabId);
    tabRefs.current.get(nextTabId)?.focus();
  };

  return (
    <div data-modal-tabs={idPrefix}>
      <div className="modal-tabs" id={`${idPrefix}-tabs`} role="tablist" aria-label={label}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            ref={(element) => {
              if (element) tabRefs.current.set(tab.id, element);
              else tabRefs.current.delete(tab.id);
            }}
            type="button"
            id={`${idPrefix}-tab-${tab.id}`}
            role="tab"
            className="modal-tab"
            aria-selected={selectedTabId === tab.id}
            aria-controls={`${idPrefix}-panel-${tab.id}`}
            tabIndex={selectedTabId === tab.id ? 0 : -1}
            onClick={() => selectTab(tab.id)}
            onKeyDown={(event) => handleTabKeyDown(event, tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          id={`${idPrefix}-panel-${tab.id}`}
          role="tabpanel"
          className={`modal-tab-panel${tab.panelClassName ? ` ${tab.panelClassName}` : ''}`}
          aria-labelledby={`${idPrefix}-tab-${tab.id}`}
          hidden={selectedTabId !== tab.id}
        >
          {visitedTabIds.has(tab.id) || selectedTabId === tab.id ? tab.content : null}
        </div>
      ))}
    </div>
  );
}
