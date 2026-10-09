import { Shield, Tag } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { ContentCategory } from '@/api/contentProtection';

export function ContentProtectionCategoryCard({
  category,
  onUpdate,
  onDelete,
}: {
  category: ContentCategory;
  onUpdate: (id: string, change: Partial<ContentCategory>) => void;
  onDelete: (category: ContentCategory) => void;
}): JSX.Element {
  const [fields, setFields] = useState(() => ({
    name: category.name,
    description: category.description,
    denial_message: category.denial_message,
    threshold_override: category.threshold_override?.toString() ?? '',
  }));
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<keyof typeof fields, string>>>({});
  const [lists, setLists] = useState(() => ({
    includes: category.includes.join('\n'),
    excludes: category.excludes.join('\n'),
    examples: category.examples.join('\n'),
  }));
  useEffect(
    () =>
      setLists({
        includes: category.includes.join('\n'),
        excludes: category.excludes.join('\n'),
        examples: category.examples.join('\n'),
      }),
    [category.examples, category.excludes, category.includes],
  );
  useEffect(() => {
    setFields({
      name: category.name,
      description: category.description,
      denial_message: category.denial_message,
      threshold_override: category.threshold_override?.toString() ?? '',
    });
  }, [category.denial_message, category.description, category.name, category.threshold_override]);
  const commitList = (field: 'includes' | 'excludes' | 'examples') =>
    onUpdate(category.id, {
      [field]: lists[field]
        .split('\n')
        .map((item) => item.trim())
        .filter(Boolean),
    });
  const updateField = (field: keyof typeof fields, value: string) => {
    setFields((current) => ({ ...current, [field]: value }));
    setFieldErrors((current) => ({ ...current, [field]: undefined }));
  };
  const cancelField = (field: keyof typeof fields) => {
    setFields((current) => ({
      ...current,
      [field]:
        field === 'threshold_override'
          ? (category.threshold_override?.toString() ?? '')
          : category[field],
    }));
    setFieldErrors((current) => ({ ...current, [field]: undefined }));
  };
  const handleFieldKeyDown = (
    field: keyof typeof fields,
    event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      cancelField(field);
    }
  };
  const commitRequiredField = (field: 'name' | 'description' | 'denial_message') => {
    if (!fields[field].trim()) {
      setFieldErrors((current) => ({ ...current, [field]: 'This field is required.' }));
      setFields((current) => ({ ...current, [field]: category[field] }));
      return;
    }
    onUpdate(category.id, { [field]: fields[field] });
  };
  const commitThreshold = () => {
    const value = fields.threshold_override;
    if (!value) {
      onUpdate(category.id, { threshold_override: null });
      return;
    }
    const threshold = Number(value);
    if (Number.isFinite(threshold) && threshold > 0 && threshold <= 1) {
      onUpdate(category.id, { threshold_override: threshold });
      return;
    }
    setFieldErrors((current) => ({
      ...current,
      threshold_override: 'Enter a threshold greater than 0 and no more than 1.',
    }));
    setFields((current) => ({
      ...current,
      threshold_override: category.threshold_override?.toString() ?? '',
    }));
  };
  return (
    <article
      className="tool-card content-protection-category"
      data-category-id={category.id}
      id={`content-protection-category-${category.id}`}
    >
      <div className="tool-card-header">
        <div className="tool-card-icon">
          {category.system ? (
            <Shield aria-hidden="true" size={28} />
          ) : (
            <Tag aria-hidden="true" size={28} />
          )}
        </div>
        <div className="tool-card-header-content">
          <div className="tool-card-header-main">
            <h3>{category.name}</h3>
            {category.system && <span className="tool-badge">System</span>}
          </div>
        </div>
      </div>
      <div className="form-group">
        <label htmlFor={`content-protection-category-${category.id}-name`}>Name</label>
        <input
          id={`content-protection-category-${category.id}-name`}
          type="text"
          disabled={category.system}
          value={fields.name}
          maxLength={128}
          aria-invalid={Boolean(fieldErrors.name)}
          aria-describedby={fieldErrors.name ? `${category.id}-name-error` : undefined}
          onChange={(event) => updateField('name', event.target.value)}
          onBlur={() => commitRequiredField('name')}
          onKeyDown={(event) => handleFieldKeyDown('name', event)}
        />
        {fieldErrors.name && (
          <p id={`${category.id}-name-error`} role="alert">
            {fieldErrors.name}
          </p>
        )}
      </div>
      <div className="form-group">
        <label htmlFor={`content-protection-category-${category.id}-description`}>
          Description
        </label>
        <textarea
          id={`content-protection-category-${category.id}-description`}
          disabled={category.system}
          value={fields.description}
          maxLength={1000}
          rows={3}
          aria-invalid={Boolean(fieldErrors.description)}
          aria-describedby={
            fieldErrors.description ? `${category.id}-description-error` : undefined
          }
          onChange={(event) => updateField('description', event.target.value)}
          onBlur={() => commitRequiredField('description')}
          onKeyDown={(event) => handleFieldKeyDown('description', event)}
        />
        {fieldErrors.description && (
          <p id={`${category.id}-description-error`} role="alert">
            {fieldErrors.description}
          </p>
        )}
      </div>
      <div className="form-group">
        <div className="access-level-field-header">
          <label htmlFor={`content-protection-category-${category.id}-denial-message`}>
            Denial message
          </label>
          <span className="field-help">{category.denial_message.length}/120</span>
        </div>
        <input
          id={`content-protection-category-${category.id}-denial-message`}
          type="text"
          disabled={category.system}
          value={fields.denial_message}
          maxLength={120}
          aria-invalid={Boolean(fieldErrors.denial_message)}
          aria-describedby={fieldErrors.denial_message ? `${category.id}-denial-error` : undefined}
          onChange={(event) => updateField('denial_message', event.target.value)}
          onBlur={() => commitRequiredField('denial_message')}
          onKeyDown={(event) => handleFieldKeyDown('denial_message', event)}
        />
        {fieldErrors.denial_message && (
          <p id={`${category.id}-denial-error`} role="alert">
            {fieldErrors.denial_message}
          </p>
        )}
      </div>
      {(['includes', 'excludes', 'examples'] as const).map((field) => (
        <div className="form-group" key={field}>
          <label htmlFor={`content-protection-category-${category.id}-${field}`}>
            {field[0].toUpperCase() + field.slice(1)}
          </label>
          <textarea
            id={`content-protection-category-${category.id}-${field}`}
            disabled={category.system}
            value={lists[field]}
            rows={2}
            onChange={(event) =>
              setLists((current) => ({ ...current, [field]: event.target.value }))
            }
            onBlur={() => commitList(field)}
          />
        </div>
      ))}
      {!category.system && (
        <label>
          Threshold override
          <input
            className="content-protection-threshold-input"
            type="number"
            min="0.01"
            max="1"
            step="0.01"
            value={fields.threshold_override}
            aria-invalid={Boolean(fieldErrors.threshold_override)}
            aria-describedby={
              fieldErrors.threshold_override ? `${category.id}-threshold-error` : undefined
            }
            onChange={(event) => updateField('threshold_override', event.target.value)}
            onBlur={commitThreshold}
            onKeyDown={(event) => handleFieldKeyDown('threshold_override', event)}
          />
          {fieldErrors.threshold_override && (
            <p id={`${category.id}-threshold-error`} role="alert">
              {fieldErrors.threshold_override}
            </p>
          )}
        </label>
      )}
      {category.id === 'rule_override' && (
        <p className="field-help">Inbound only. Cannot be granted.</p>
      )}
      {!category.system && (
        <div className="tool-card-footer">
          <div className="tool-card-actions">
            <button
              type="button"
              className="btn btn-sm btn-danger"
              onClick={() => onDelete(category)}
            >
              Delete
            </button>
          </div>
        </div>
      )}
    </article>
  );
}
