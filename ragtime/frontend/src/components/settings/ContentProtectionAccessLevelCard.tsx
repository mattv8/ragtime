import { Lock } from 'lucide-react';
import type { AccessLevel, ContentCategory } from '@/api/contentProtection';
export function ContentProtectionAccessLevelCard({
  level,
  categories,
  disabled,
  fieldsDisabled,
  deleteDisabledReason,
  idPrefix,
  hideHeader,
  onUpdate,
  onDelete,
}: {
  level: AccessLevel;
  categories: ContentCategory[];
  disabled?: boolean;
  fieldsDisabled?: boolean;
  deleteDisabledReason?: string;
  idPrefix?: string;
  hideHeader?: boolean;
  onUpdate: (id: string, change: Partial<AccessLevel>) => void;
  onDelete?: (level: AccessLevel) => void;
}): JSX.Element {
  const descriptionIdPrefix = `${idPrefix ?? 'content-protection-'}level-${level.id}-`;
  const toggleCategory = (categoryId: string, checked: boolean) =>
    onUpdate(level.id, {
      granted_category_ids: checked
        ? [...level.granted_category_ids, categoryId]
        : level.granted_category_ids.filter((id) => id !== categoryId),
    });
  return (
    <article
      className="tool-card content-protection-level"
      data-level-id={level.id}
      id={`${idPrefix ?? 'content-protection-'}level-${level.id}`}
    >
      {!hideHeader && (
        <div className="tool-card-header">
          <div className="tool-card-icon">
            <Lock aria-hidden="true" size={28} />
          </div>
          <div className="tool-card-header-content">
            <h3>{level.name}</h3>
          </div>
        </div>
      )}
      <div className="form-group">
        <label htmlFor={`${descriptionIdPrefix}name`}>Name</label>
        <input
          id={`${descriptionIdPrefix}name`}
          type="text"
          value={level.name}
          maxLength={128}
          disabled={fieldsDisabled}
          onChange={(event) => onUpdate(level.id, { name: event.target.value })}
        />
      </div>
      <fieldset className="content-protection-fieldset">
        <legend className="access-level-section-heading">Granted categories</legend>
        <div className="content-protection-level-grants">
          {categories
            .filter((category) => category.id !== 'rule_override')
            .map((category) => (
              <label className="checkbox-label" key={category.id}>
                <input
                  className="access-level-category-checkbox"
                  aria-label={category.name}
                  aria-describedby={`${descriptionIdPrefix}category-${category.id}-description`}
                  type="checkbox"
                  checked={level.granted_category_ids.includes(category.id)}
                  onChange={(event) => toggleCategory(category.id, event.target.checked)}
                  disabled={fieldsDisabled}
                />
                <span className="checkbox-label-text">{category.name}</span>
                <span
                  id={`${descriptionIdPrefix}category-${category.id}-description`}
                  className="field-help"
                >
                  {category.description}
                </span>
              </label>
            ))}
        </div>
      </fieldset>
      <div className="form-group">
        <div className="access-level-field-header">
          <label htmlFor={`${descriptionIdPrefix}guidance`}>Guidance for assistant</label>
          <span className="field-help">{level.guidance.length}/4000</span>
        </div>
        <textarea
          id={`${descriptionIdPrefix}guidance`}
          rows={4}
          maxLength={4000}
          value={level.guidance}
          disabled={fieldsDisabled}
          onChange={(event) => onUpdate(level.id, { guidance: event.target.value })}
        />
      </div>
      {onDelete !== undefined && (
        <div className="tool-card-footer">
          <div className="tool-card-actions">
            <button
              type="button"
              className="btn btn-sm btn-danger"
              disabled={disabled}
              title={
                disabled
                  ? deleteDisabledReason || 'This access level cannot be deleted.'
                  : undefined
              }
              onClick={() => onDelete(level)}
            >
              Delete
            </button>
            {disabled && (
              <p className="field-help">
                {deleteDisabledReason ||
                  'Reassign the default level and remove group mappings before deleting.'}
              </p>
            )}
          </div>
        </div>
      )}
    </article>
  );
}
