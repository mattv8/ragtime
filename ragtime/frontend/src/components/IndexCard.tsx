import { ReactNode, useState, useRef, useEffect } from 'react';
import { Icon } from './Icon';

interface IndexCardProps {
  title: string;
  description?: string;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  onEditTitle?: (newTitle: string) => Promise<void>;
  onEditDescription?: (newDescription: string) => Promise<void>;
  metaPills?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode; // For other content in the info area
  className?: string;
  as?: 'li' | 'div';
  id?: string;
  toggleTitle?: string;
  titleChildren?: ReactNode; // Elements to render alongside the title
}

export function IndexCard({
  title,
  description,
  enabled,
  onToggle,
  onEditTitle,
  onEditDescription,
  metaPills,
  actions,
  children,
  className = '',
  as = 'div',
  id,
  toggleTitle,
  titleChildren,
}: IndexCardProps) {
  const Component = as;

  // Inline editing state
  const [editingField, setEditingField] = useState<'title' | 'description' | null>(null);
  const [editValue, setEditValue] = useState('');
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  const autoResizeTextarea = (element: HTMLTextAreaElement | null) => {
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${element.scrollHeight}px`;
  };

  // Focus and setup inputs
  useEffect(() => {
    if (editingField === 'title' && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    } else if (editingField === 'description' && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.select();
      autoResizeTextarea(textareaRef.current);
    }
  }, [editingField]);

  const handleStartEdit = (field: 'title' | 'description') => {
    if (field === 'title') {
      setEditValue(title);
    } else {
      setEditValue(description || '');
    }
    setEditingField(field);
  };

  const handleCancelEdit = () => {
    setEditingField(null);
    setEditValue('');
  };

  const handleSaveEdit = async () => {
    if (saving) return;

    // Don't save if unchanged
    if (editingField === 'title' && editValue === title) {
      handleCancelEdit();
      return;
    }
    if (editingField === 'description' && editValue === (description || '')) {
      handleCancelEdit();
      return;
    }

    // Don't allow empty title
    if (editingField === 'title' && !editValue.trim()) {
      handleCancelEdit();
      return;
    }

    setSaving(true);
    try {
      if (editingField === 'title' && onEditTitle) {
        await onEditTitle(editValue.trim());
      } else if (editingField === 'description' && onEditDescription) {
        await onEditDescription(editValue);
      }
      setEditingField(null);
    } catch (err) {
      console.error('Failed to save:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleCancelEdit();
    } else if (e.key === 'Enter') {
      // For textarea, require Ctrl+Enter to save (allow newlines)
      if (editingField === 'title' || e.ctrlKey) {
        e.preventDefault();
        handleSaveEdit();
      }
    }
  };

  return (
    <Component
      className={`index-item ${!enabled ? 'index-disabled' : ''} ${className}`}
      id={id}
    >
      <div className="index-info">
        {/* Header Row: Title and Actions */}
        <div className="index-title-row">
          <div className="index-title-container">
            {editingField === 'title' ? (
              <div className="inline-edit-field title-edit">
                <input
                  ref={inputRef}
                  type="text"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onBlur={handleSaveEdit}
                  disabled={saving}
                  className="inline-edit-input"
                />
              </div>
            ) : (
              <div
                className={`editable-field-wrapper name-wrapper ${onEditTitle ? 'editable' : ''}`}
                onClick={onEditTitle ? () => handleStartEdit('title') : undefined}
              >
                <h3>
                  {title}
                  {titleChildren}
                </h3>
                {onEditTitle && (
                  <button
                    type="button"
                    className="inline-edit-btn"
                    onClick={(e) => { e.stopPropagation(); handleStartEdit('title'); }}
                    title="Edit name"
                  >
                    <Icon name="pencil" size={12} />
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Actions moved here */}
          <div className="index-actions">
             {actions}
             <div className="index-toggle">
                <label
                  className="toggle-switch"
                  title={toggleTitle || (enabled ? 'Enabled' : 'Disabled')}
                >
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={(e) => onToggle(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
             </div>
          </div>
        </div>

        {/* Description Row - Full Width */}
        {(description !== undefined || onEditDescription) && (
          <div className="index-description-row">
            {editingField === 'description' ? (
              <div className="inline-edit-field description-edit">
                <textarea
                  ref={textareaRef}
                  value={editValue}
                  onChange={(e) => {
                    setEditValue(e.target.value);
                    autoResizeTextarea(e.target);
                  }}
                  onKeyDown={handleKeyDown}
                  onBlur={handleSaveEdit}
                  disabled={saving}
                  className="inline-edit-textarea"
                  rows={3}
                  placeholder="Description for AI..."
                />
              </div>
            ) : (
              <div
                className={`editable-field-wrapper description-wrapper ${onEditDescription ? 'editable' : ''}`}
                onClick={onEditDescription ? () => handleStartEdit('description') : undefined}
              >
                {description ? (
                  <p className="index-description truncated" title="Click to edit and view full description">
                    {description}
                  </p>
                ) : (
                  onEditDescription && <p className="index-description placeholder">Add description for AI...</p>
                )}
                {onEditDescription && (
                  <button
                    type="button"
                    className="inline-edit-btn"
                    onClick={(e) => { e.stopPropagation(); handleStartEdit('description'); }}
                    title="Edit description"
                  >
                    <Icon name="pencil" size={12} />
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {metaPills && <div className="index-meta-pills">{metaPills}</div>}
        {children}
      </div>

    </Component>
  );
}
