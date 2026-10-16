import { Shield, Tag } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { ContentCategory } from '@/api/contentProtection';
import { InlineEditableCardTitle } from './InlineEditableCardTitle';

const STRICTNESS_PRESETS = {
  strict: 0.25,
  balanced: 0.5,
  permissive: 0.75,
} as const;
const STRICTNESS_LABELS = {
  strict: 'Strict',
  balanced: 'Balanced',
  permissive: 'Permissive',
} as const;

type Strictness = keyof typeof STRICTNESS_PRESETS;

const formatPercentage = (value: number) => String(Number((value * 100).toPrecision(12)));

export const categoryStrictnessLabel = (threshold: number) => {
  const preset = Object.entries(STRICTNESS_PRESETS).find(([, value]) => value === threshold)?.[0];
  return preset
    ? `${STRICTNESS_LABELS[preset as Strictness]} (≥${formatPercentage(threshold)}%)`
    : `Custom (≥${formatPercentage(threshold)}%)`;
};

export function ContentProtectionCategoryCard({
  category,
  globalStrictness,
  hideHeader,
  onUpdate,
}: {
  category: ContentCategory;
  globalStrictness: Strictness;
  hideHeader?: boolean;
  onUpdate: (id: string, change: Partial<ContentCategory>) => void;
}): JSX.Element {
  const [fields, setFields] = useState(() => ({
    description: category.description,
    denial_message: category.denial_message,
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
      description: category.description,
      denial_message: category.denial_message,
    });
  }, [category.denial_message, category.description, category.threshold_override]);
  const commitList = (field: 'includes' | 'excludes' | 'examples', value = lists[field]) =>
    onUpdate(category.id, {
      [field]: value
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
      [field]: category[field],
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
  const commitRequiredField = (field: 'description' | 'denial_message') => {
    if (!fields[field].trim()) {
      setFieldErrors((current) => ({ ...current, [field]: 'This field is required.' }));
      setFields((current) => ({ ...current, [field]: category[field] }));
      return;
    }
    onUpdate(category.id, { [field]: fields[field] });
  };
  const preset = Object.entries(STRICTNESS_PRESETS).find(
    ([, value]) => value === category.threshold_override,
  )?.[0] as Strictness | undefined;
  const isCustom = category.threshold_override !== null && !preset;
  const strictnessValue = category.threshold_override === null ? '' : (preset ?? 'custom');
  const globalPercentage = formatPercentage(STRICTNESS_PRESETS[globalStrictness]);
  return (
    <article
      className="tool-card content-protection-category"
      data-category-id={category.id}
      id={`content-protection-category-${category.id}`}
    >
      {!hideHeader && (
        <div className="tool-card-header">
          <div className="tool-card-icon">
            {category.system ? (
              <Shield aria-hidden="true" size={20} />
            ) : (
              <Tag aria-hidden="true" size={20} />
            )}
          </div>
          <div className="tool-card-header-content">
            <div className="tool-card-header-main">
              <InlineEditableCardTitle
                value={category.name}
                ariaLabel="Category name"
                maxLength={128}
                disabled={category.system}
                onCommit={(name) => onUpdate(category.id, { name })}
              />
              {category.system && <span className="tool-badge">System</span>}
            </div>
          </div>
        </div>
      )}
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
            onKeyDown={(event) => {
              if (event.key === 'Escape') commitList(field, event.currentTarget.value);
            }}
          />
        </div>
      ))}
      {!category.system && (
        <div className="form-group">
          <label htmlFor={`content-protection-category-${category.id}-strictness`}>
            Strictness
          </label>
          <select
            id={`content-protection-category-${category.id}-strictness`}
            value={strictnessValue}
            onChange={(event) => {
              const value = event.target.value as Strictness | '' | 'custom';
              if (value === '') onUpdate(category.id, { threshold_override: null });
              if (value in STRICTNESS_PRESETS)
                onUpdate(category.id, {
                  threshold_override: STRICTNESS_PRESETS[value as Strictness],
                });
            }}
          >
            <option value="">
              Follow global — {STRICTNESS_LABELS[globalStrictness]} (≥{globalPercentage}%)
            </option>
            {Object.entries(STRICTNESS_PRESETS).map(([value, threshold]) => (
              <option key={value} value={value}>
                {categoryStrictnessLabel(threshold)}
              </option>
            ))}
            {isCustom && (
              <option value="custom">
                {categoryStrictnessLabel(category.threshold_override!)}
              </option>
            )}
          </select>
          <p className="field-help">
            Denied at or above this threshold only for an audience without a grant. Lower thresholds
            block more.
          </p>
        </div>
      )}
      {category.id === 'rule_override' && (
        <p className="field-help">Inbound only. Cannot be granted.</p>
      )}
    </article>
  );
}
