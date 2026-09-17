import { useEffect, useRef, useState } from 'react';
import type { AuthProviderConfig } from '@/types';

interface TimeoutField {
  key: 'web_session_hours' | 'mcp_access_token_minutes' | 'mcp_authorization_days';
  label: string;
  help: string;
  min: number;
  max: number;
}

const TIMEOUT_FIELDS: TimeoutField[] = [
  {
    key: 'web_session_hours',
    label: 'Web session override (hours)',
    help: 'Fixed lifetime; activity does not extend it. Leave blank to inherit JWT_EXPIRE_HOURS.',
    min: 1,
    max: 720,
  },
  {
    key: 'mcp_access_token_minutes',
    label: 'MCP access token lifetime (minutes)',
    help: 'Fixed lifetime for interactive MCP access tokens, capped by their authorization grant.',
    min: 5,
    max: 1440,
  },
  {
    key: 'mcp_authorization_days',
    label: 'MCP authorization lifetime (days)',
    help: 'Interactive MCP authorization grants expire after this absolute lifetime.',
    min: 1,
    max: 90,
  },
];

function getDrafts(config: AuthProviderConfig): Record<TimeoutField['key'], string> {
  return {
    web_session_hours: config.web_session_hours == null ? '' : String(config.web_session_hours),
    mcp_access_token_minutes: String(config.mcp_access_token_minutes ?? 60),
    mcp_authorization_days: String(config.mcp_authorization_days ?? 30),
  };
}

function validate(field: TimeoutField, value: string): string | null {
  if (field.key === 'web_session_hours' && value.trim() === '') {
    return null;
  }
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value))) {
    return `Enter a whole number from ${field.min} to ${field.max}.`;
  }
  const numericValue = Number(value);
  return numericValue >= field.min && numericValue <= field.max
    ? null
    : `Enter a whole number from ${field.min} to ${field.max}.`;
}

interface AuthenticationTimeoutsProps {
  config: AuthProviderConfig;
  onChange: (next: AuthProviderConfig) => void;
  onValidityChange: (valid: boolean) => void;
}

export function AuthenticationTimeouts({
  config,
  onChange,
  onValidityChange,
}: AuthenticationTimeoutsProps) {
  const [drafts, setDrafts] = useState(() => getDrafts(config));
  const [errors, setErrors] = useState<Record<TimeoutField['key'], string | null>>({
    web_session_hours: null,
    mcp_access_token_minutes: null,
    mcp_authorization_days: null,
  });
  const previousConfigRef = useRef(config);

  useEffect(() => {
    if (previousConfigRef.current === config) {
      return;
    }
    previousConfigRef.current = config;
    const serverDrafts = getDrafts(config);
    setDrafts((current) => {
      const next = { ...current };
      for (const field of TIMEOUT_FIELDS) {
        if (!errors[field.key]) {
          next[field.key] = serverDrafts[field.key];
        }
      }
      return next;
    });
  }, [config, errors]);

  useEffect(() => {
    onValidityChange(!Object.values(errors).some(Boolean));
  }, [errors, onValidityChange]);

  const updateField = (field: TimeoutField, value: string) => {
    const error = validate(field, value);
    const nextErrors = { ...errors, [field.key]: error };
    setDrafts((current) => ({ ...current, [field.key]: value }));
    setErrors(nextErrors);
    if (error) {
      return;
    }
    onChange({
      ...config,
      [field.key]: field.key === 'web_session_hours' && value.trim() === '' ? null : Number(value),
    });
  };

  return (
    <div id="authentication-timeout-policy" className="form-group">
      <h4>Session and MCP token lifetimes</h4>
      <p className="fieldset-help auth-provider-help-tight">
        These controls apply to web sessions and interactive MCP OAuth. Client-credentials tokens
        remain one hour.
      </p>
      <div className="form-row">
        {TIMEOUT_FIELDS.map((field) => {
          const id = `authentication-${field.key}`;
          return (
            <div className="form-group" key={field.key}>
              <label htmlFor={id}>{field.label}</label>
              <input
                id={id}
                type="number"
                min={field.min}
                max={field.max}
                step="1"
                value={drafts[field.key]}
                aria-describedby={`${id}-help${errors[field.key] ? ` ${id}-error` : ''}`}
                aria-invalid={Boolean(errors[field.key])}
                onChange={(event) => updateField(field, event.target.value)}
              />
              <p id={`${id}-help`} className="field-help">
                {field.help}
              </p>
              {field.key === 'web_session_hours' && (
                <>
                  <p className="field-help">
                    Effective web session lifetime:{' '}
                    {config.effective_web_session_hours ?? 'unavailable'} hours.
                  </p>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={drafts.web_session_hours === ''}
                    onClick={() => updateField(field, '')}
                  >
                    Reset to inherited value
                  </button>
                </>
              )}
              {errors[field.key] && (
                <p id={`${id}-error`} className="field-error" role="alert">
                  {errors[field.key]}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
