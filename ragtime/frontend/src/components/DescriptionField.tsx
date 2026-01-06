import { useEffect, useState } from 'react';
import { api } from '@/api';

interface DescriptionFieldProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  rows?: number;
  /** If true, show in compact mode with smaller text and less padding */
  compact?: boolean;
}

/**
 * Reusable description field for index creation/editing.
 * Automatically detects if AI auto-generation is available based on configured API keys.
 */
export function DescriptionField({
  value,
  onChange,
  disabled = false,
  rows = 3,
  compact = false,
}: DescriptionFieldProps) {
  const [hasLlmKey, setHasLlmKey] = useState<boolean | null>(null);

  useEffect(() => {
    // Check if user has configured an LLM API key for auto-generation
    const checkSettings = async () => {
      try {
        const settings = await api.getSettings();
        const hasKey = !!(settings.openai_api_key || settings.anthropic_api_key);
        setHasLlmKey(hasKey);
      } catch {
        setHasLlmKey(false);
      }
    };
    checkSettings();
  }, []);

  const placeholder = hasLlmKey
    ? 'Describe what this index contains, or leave blank to auto-generate'
    : 'Describe what this index contains for AI context';

  const helpText = hasLlmKey
    ? 'Helps the AI understand when to search this index. Leave blank to auto-generate from content.'
    : 'Helps the AI understand when to search this index.';

  return (
    <div className="form-group">
      <label>Description</label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        style={{
          width: '100%',
          resize: 'vertical',
          minHeight: compact ? '60px' : '80px',
        }}
        disabled={disabled}
      />
      <small style={{ color: '#888', fontSize: '0.8rem' }}>
        {helpText}
      </small>
    </div>
  );
}
