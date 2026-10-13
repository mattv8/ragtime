import type { ReactNode } from 'react';

export interface AccessLevelGroup {
  id: string;
  label: string;
  badge?: ReactNode;
}

interface AccessLevelGroupsFieldProps {
  idPrefix: string;
  groups: AccessLevelGroup[];
  mappedIds: string[];
  onAdd: (id: string) => void;
  onRemove: (id: string) => void;
  disabled?: boolean;
  disabledHint?: ReactNode;
}

export function AccessLevelGroupsField({
  idPrefix,
  groups,
  mappedIds,
  onAdd,
  onRemove,
  disabled = false,
  disabledHint,
}: AccessLevelGroupsFieldProps): JSX.Element {
  return (
    <fieldset
      id={`${idPrefix}groups`}
      className="content-protection-fieldset"
      data-level-section="groups"
    >
      <legend className="access-level-section-heading">Groups</legend>
      <div className="access-level-groups-chips" data-level-groups-chips>
        {mappedIds.map((id) => {
          const group = groups.find((candidate) => candidate.id === id);
          const isUnknown = !group;

          return (
            <span
              key={id}
              className={`access-level-group-chip${isUnknown ? ' access-level-group-chip-unknown' : ''}`}
              data-level-group-chip={id}
            >
              {group?.badge}
              {group ? group.label : `Unknown (${id.slice(0, 8)})`}
              <button
                type="button"
                className="access-level-group-chip-remove"
                aria-label={
                  group ? `Remove ${group.label} from this level` : `Remove unknown group ${id}`
                }
                disabled={disabled}
                onClick={() => onRemove(id)}
              >
                ×
              </button>
            </span>
          );
        })}
      </div>
      {disabled && disabledHint ? (
        <p className="field-help">{disabledHint}</p>
      ) : (
        <select
          aria-label="Add group to this level"
          data-add-group-select
          value=""
          disabled={disabled}
          onChange={(event) => {
            if (event.target.value) onAdd(event.target.value);
          }}
        >
          <option value="">Add a group…</option>
          {groups
            .filter((group) => !mappedIds.includes(group.id))
            .map((group) => (
              <option key={group.id} value={group.id}>
                {group.label}
              </option>
            ))}
        </select>
      )}
    </fieldset>
  );
}
