import type { ReactNode } from 'react';
import type { AccessLevel, ContentCategory } from '@/api/contentProtection';
import { ContentProtectionAccessLevelCard } from '../settings/ContentProtectionAccessLevelCard';
import { AccessDetailHeader } from './AccessDetailHeader';
import { AccessLevelGroupsField, type AccessLevelGroup } from './AccessLevelGroupsField';
import { MasterDetailLayout, type MasterDetailLayoutProps } from './MasterDetailLayout';
import { RowDeleteButton } from './RowDeleteButton';

type PassThroughLayoutProps = Pick<
  MasterDetailLayoutProps,
  | 'search'
  | 'emptyMessage'
  | 'noMatchMessage'
  | 'emptyHookAttribute'
  | 'noMatchHookAttribute'
  | 'rootProps'
  | 'listProps'
  | 'listBodyProps'
  | 'detailProps'
  | 'rowHookAttribute'
>;

export interface AccessLevelsMasterDetailProps extends PassThroughLayoutProps {
  id: string;
  /** Prefix for detail field ids so several editors can coexist in the document. */
  idPrefix: string;
  headingId: string;
  levels: AccessLevel[];
  selectedLevel?: AccessLevel;
  categories: ContentCategory[];
  groups: AccessLevelGroup[];
  defaultLevelId: string;
  groupIdsForLevel: (levelId: string) => string[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCloseDetail: () => void;
  onAdd: () => void;
  onUpdate: (id: string, change: Partial<AccessLevel>) => void;
  onAddGroup: (levelId: string, groupId: string) => void;
  onRemoveGroup: (levelId: string, groupId: string) => void;
  onMakeDefault: (levelId: string) => void;
  onDelete: (level: AccessLevel) => boolean | Promise<boolean>;
  deleteBlockedReason: (level: AccessLevel) => string | undefined;
  confirmDelete?: boolean;
  busy?: boolean;
  fieldsDisabled?: boolean;
  levelActionsDisabled?: boolean;
  groupsDisabledHint?: ReactNode;
  startEditingLevelId?: string | null;
  rowIndicator?: (level: AccessLevel) => ReactNode;
  /** Rendered at the top of the detail body, above the level fields. */
  detailIntro?: ReactNode;
  /** Rendered at the bottom of the detail body, below the default-level section. */
  detailExtras?: ReactNode;
  /** Rendered after the scrolling detail body (banners, footers). */
  detailFooter?: ReactNode;
}

function pluralize(count: number, singular: string, plural: string): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

/** Why a saved access level cannot be deleted yet, shared so both editors explain it identically. */
export function accessLevelDeleteBlockedReason(
  isDefault: boolean,
  mappedGroupCount: number,
): string | undefined {
  if (isDefault) return 'Cannot delete the default level. Assign a different default first.';
  if (mappedGroupCount)
    return `Mapped to ${pluralize(mappedGroupCount, 'group', 'groups')}. Remove all group mappings before deleting.`;
  return undefined;
}

function DefaultBadge() {
  return <span className="access-level-default-badge">Default</span>;
}

export function AccessLevelsMasterDetail({
  id,
  idPrefix,
  headingId,
  levels,
  selectedLevel,
  categories,
  groups,
  defaultLevelId,
  groupIdsForLevel,
  selectedId,
  onSelect,
  onCloseDetail,
  onAdd,
  onUpdate,
  onAddGroup,
  onRemoveGroup,
  onMakeDefault,
  onDelete,
  deleteBlockedReason,
  confirmDelete = false,
  busy = false,
  fieldsDisabled = false,
  levelActionsDisabled = false,
  groupsDisabledHint,
  startEditingLevelId,
  rowIndicator,
  detailIntro,
  detailExtras,
  detailFooter,
  emptyMessage = 'No access levels configured.',
  ...layoutProps
}: AccessLevelsMasterDetailProps): JSX.Element {
  const items = levels.map((level) => {
    const groupCount = groupIdsForLevel(level.id).length;
    const categoryCount = level.granted_category_ids.length;
    return {
      id: level.id,
      name: level.name || '(new level)',
      meta: (
        <>
          {level.id === defaultLevelId && <DefaultBadge />}
          {groupCount > 0 && <span>{pluralize(groupCount, 'group', 'groups')}</span>}
          {categoryCount > 0 && <span>{pluralize(categoryCount, 'category', 'categories')}</span>}
          {rowIndicator?.(level)}
        </>
      ),
      action: (
        <RowDeleteButton
          label={`Delete ${level.name || 'access level'}`}
          blockedReason={deleteBlockedReason(level)}
          disabled={busy}
          confirm={confirmDelete}
          onDelete={() => onDelete(level)}
        />
      ),
    };
  });

  const levelIdPrefix = selectedLevel ? `${idPrefix}${selectedLevel.id}-` : idPrefix;
  const isDefault = selectedLevel?.id === defaultLevelId;

  return (
    <MasterDetailLayout
      {...layoutProps}
      id={id}
      heading="Access levels"
      headerAction={
        <button
          className="btn btn-secondary btn-sm"
          type="button"
          data-new-access-level
          disabled={busy}
          onClick={onAdd}
        >
          Add access level
        </button>
      }
      items={items}
      selectedId={selectedId}
      onSelect={onSelect}
      onCloseDetail={onCloseDetail}
      detailLabelledBy={headingId}
      emptyMessage={emptyMessage}
      busy={busy}
      detail={
        selectedLevel && (
          <>
            <AccessDetailHeader
              key={selectedLevel.id}
              headingId={headingId}
              title={selectedLevel.name}
              meta={isDefault ? <DefaultBadge /> : undefined}
              editableTitle={{
                ariaLabel: 'Access level name',
                maxLength: 128,
                disabled: fieldsDisabled,
                startEditing: startEditingLevelId === selectedLevel.id,
                onCommit: (name) => onUpdate(selectedLevel.id, { name }),
              }}
              closeLabel="Close access level editor"
              onClose={onCloseDetail}
            />
            <div className="access-md-detail-body" data-level-detail-body>
              {detailIntro}
              <ContentProtectionAccessLevelCard
                key={selectedLevel.id}
                level={selectedLevel}
                categories={categories}
                idPrefix={levelIdPrefix}
                hideHeader
                fieldsDisabled={fieldsDisabled}
                onUpdate={onUpdate}
              />
              <AccessLevelGroupsField
                idPrefix={levelIdPrefix}
                groups={groups}
                mappedIds={groupIdsForLevel(selectedLevel.id)}
                disabled={levelActionsDisabled}
                disabledHint={groupsDisabledHint}
                onAdd={(groupId) => onAddGroup(selectedLevel.id, groupId)}
                onRemove={(groupId) => onRemoveGroup(selectedLevel.id, groupId)}
              />
              <section className="content-protection-fieldset" data-level-section="default">
                <h5 className="access-level-section-heading">Default level</h5>
                <p className="field-help">
                  All verified users with no group mapping receive the default level.
                </p>
                {!isDefault && (
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    data-make-default-btn
                    disabled={levelActionsDisabled}
                    onClick={() => onMakeDefault(selectedLevel.id)}
                  >
                    Make default level
                  </button>
                )}
              </section>
              {detailExtras}
            </div>
            {detailFooter}
          </>
        )
      }
    />
  );
}
