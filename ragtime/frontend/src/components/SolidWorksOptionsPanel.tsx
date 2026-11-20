import { useEffect, useState, type ReactNode } from 'react';
import { AlertTriangle, Info, XCircle, type LucideIcon } from 'lucide-react';
import {
  SOLIDWORKS_DEFAULT_OPTIONS,
  type SolidworksHostStatus,
  type SolidworksInstanceMode,
  type SolidworksOptions,
} from '../api/solidworks';

export interface SolidWorksOptionsPanelProps {
  value: SolidworksOptions;
  onChange: (value: SolidworksOptions) => void;
  hostStatus: SolidworksHostStatus | null;
  timeoutMaxSeconds: number;
  onTimeoutMaxSecondsChange: (seconds: number) => void;
  disabled?: boolean;
}

type Tone = 'info' | 'warning' | 'error';
const TONE_ICON: Record<Tone, LucideIcon> = { info: Info, warning: AlertTriangle, error: XCircle };
const TIMEOUT_MIN = 15;
const TIMEOUT_MAX = 3600;
const TIMEOUT_SLIDER = { min: 15, max: 300, step: 5 } as const;
const TIMEOUT_PROXY_WARNING_ABOVE = 110;
const IDLE_MINUTES_MAX = 1440;
const MAX_INSTANCES_LIMIT = 4;

const INSTANCE_MODES: ReadonlyArray<readonly [SolidworksInstanceMode, string, string]> = [
  [
    'background',
    'Background instance',
    'The host starts its own hidden SolidWorks and closes it when idle. Best for a shared or unattended machine.',
  ],
  [
    'attach',
    'Attach to open SolidWorks',
    'Reads from the SolidWorks the user already has open. Only documents that are already open are available, and it never saves, closes or kills anything.',
  ],
];

function Message({ id, tone, children }: { id: string; tone: Tone; children: ReactNode }) {
  const Icon = TONE_ICON[tone];
  return (
    <p id={id} className="sw-message" data-tone={tone}>
      <Icon className="sw-message-icon" size={14} aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}

function parseDraft(text: string): number {
  return text.trim() === '' ? Number.NaN : Number(text);
}

function isWholeInRange(value: number, min: number, max: number): boolean {
  return Number.isInteger(value) && value >= min && value <= max;
}

interface NumberFieldProps {
  inputId: string;
  label: string;
  value: number;
  min: number;
  max: number;
  helpId: string;
  help: string;
  errorId: string;
  /** Receives the parsed draft (NaN when empty or not a number). */
  error: (draft: number) => string;
  warningId?: string;
  warning?: string;
  slider?: { id: string };
  onChange: (value: number) => void;
}

function NumberField({
  inputId,
  label,
  value,
  min,
  max,
  helpId,
  help,
  errorId,
  error,
  warningId,
  warning,
  slider,
  onChange,
}: NumberFieldProps) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const parsed = parseDraft(draft);
  const valid = isWholeInRange(parsed, min, max);
  const describedBy = [helpId, !valid && errorId, warning && warningId].filter(Boolean).join(' ');

  const change = (text: string) => {
    setDraft(text);
    const next = parseDraft(text);
    if (isWholeInRange(next, min, max)) onChange(next);
  };
  const blur = () => {
    if (valid) return;
    const next = Number.isFinite(parsed) ? Math.min(max, Math.max(min, Math.round(parsed))) : value;
    setDraft(String(next));
    onChange(next);
  };

  return (
    <div className="form-group">
      <label htmlFor={inputId}>{label}</label>
      <div className={slider ? 'sw-timeout-control' : undefined}>
        {slider && (
          <input
            id={slider.id}
            type="range"
            min={TIMEOUT_SLIDER.min}
            max={TIMEOUT_SLIDER.max}
            step={TIMEOUT_SLIDER.step}
            value={Math.min(TIMEOUT_SLIDER.max, Math.max(TIMEOUT_SLIDER.min, value))}
            aria-label={`${label} slider`}
            onChange={(event) => change(event.target.value)}
          />
        )}
        <input
          id={inputId}
          type="number"
          min={min}
          max={max}
          step={1}
          value={draft}
          onChange={(event) => change(event.target.value)}
          onBlur={blur}
          aria-invalid={!valid}
          aria-describedby={describedBy}
        />
      </div>
      <p id={helpId} className="field-help">
        {help}
      </p>
      {!valid && (
        <Message id={errorId} tone="error">
          {error(parsed)}
        </Message>
      )}
      {warning && warningId && (
        <Message id={warningId} tone="warning">
          {warning}
        </Message>
      )}
    </div>
  );
}

function timeoutError(draft: number): string {
  if (draft < TIMEOUT_MIN) return `Enter at least ${TIMEOUT_MIN} seconds.`;
  if (draft > TIMEOUT_MAX) return `Enter at most ${TIMEOUT_MAX} seconds.`;
  return `Enter a whole number of seconds from ${TIMEOUT_MIN} to ${TIMEOUT_MAX}.`;
}

function versionHelp(pin: number | null, mode: SolidworksInstanceMode): string {
  if (pin === null) return 'Uses the newest SolidWorks found on the machine.';
  return mode === 'background'
    ? `The host only starts SolidWorks ${pin}. Calls fail if it is not installed.`
    : `Calls fail if the open SolidWorks is not ${pin}.`;
}

export function SolidWorksOptionsPanel({
  value,
  onChange,
  hostStatus,
  timeoutMaxSeconds,
  onTimeoutMaxSecondsChange,
  disabled,
}: SolidWorksOptionsPanelProps) {
  const options = { ...SOLIDWORKS_DEFAULT_OPTIONS, ...value };
  const update = (patch: SolidworksOptions) => onChange({ ...options, ...patch });

  const versions = [...(hostStatus?.solidworks?.installed_versions ?? [])].sort((a, b) => b - a);
  const known = hostStatus?.running === true;
  const pin = options.solidworks_version ?? null;
  const pinNotInstalled = known && pin !== null && !versions.includes(pin);
  const versionMessage = !known
    ? {
        id: 'solidworks-version-warning',
        tone: 'info' as const,
        text: 'Install and start the host to choose from the versions found on the machine.',
      }
    : pinNotInstalled
      ? {
          id: 'solidworks-version-warning',
          tone: 'warning' as const,
          text:
            versions.length > 0
              ? `SolidWorks ${pin} was not found on this machine (found: ${versions.join(', ')}). Calls fail until you install it or choose another version.`
              : `SolidWorks ${pin} was not found: no SolidWorks installation is visible to this account.`,
        }
      : null;
  const versionDescribedBy = ['solidworks-version-help', versionMessage?.id]
    .filter(Boolean)
    .join(' ');
  const background = options.instance_mode === 'background';
  const timeoutWarning =
    timeoutMaxSeconds > TIMEOUT_PROXY_WARNING_ABOVE
      ? 'Calls running past about 120 seconds may be cut off by the web proxy before the host can answer.'
      : undefined;
  const timeoutField = (
    <NumberField
      inputId="solidworks-timeout-max-seconds"
      label="Max Timeout (seconds)"
      value={timeoutMaxSeconds}
      min={TIMEOUT_MIN}
      max={TIMEOUT_MAX}
      helpId="solidworks-timeout-help"
      help="Longest a single SolidWorks call may take, including starting SolidWorks (about 10 seconds) and opening large assemblies. The default is 100 because the web proxy in front of Ragtime ends requests at about 120 seconds."
      errorId="solidworks-timeout-error"
      error={timeoutError}
      warningId="solidworks-timeout-warning"
      warning={timeoutWarning}
      slider={{ id: 'solidworks-timeout-range' }}
      onChange={onTimeoutMaxSecondsChange}
    />
  );

  return (
    <fieldset
      id="solidworks-options-panel"
      className="sw-options"
      data-testid="solidworks-options-panel"
      disabled={disabled}
    >
      <legend>SolidWorks options</legend>
      <div
        id="solidworks-instance-mode"
        role="radiogroup"
        aria-labelledby="solidworks-instance-mode-label"
        className="sw-choice-grid"
      >
        <span id="solidworks-instance-mode-label" className="sw-group-label">
          Instance mode
        </span>
        {INSTANCE_MODES.map(([mode, title, description]) => (
          <label className="sw-choice-card" key={mode}>
            <input
              id={`solidworks-instance-mode-${mode}`}
              type="radio"
              name="solidworks-instance-mode"
              value={mode}
              checked={options.instance_mode === mode}
              onChange={() => update({ instance_mode: mode })}
            />
            <span>
              <strong>{title}</strong>
              <small>{description}</small>
            </span>
          </label>
        ))}
      </div>
      <div className="form-row">
        <div className="form-group">
          <label htmlFor="solidworks-version">SolidWorks version</label>
          <select
            id="solidworks-version"
            value={pin === null ? '' : String(pin)}
            aria-describedby={versionDescribedBy}
            onChange={(event) =>
              update({
                solidworks_version: event.target.value === '' ? null : Number(event.target.value),
              })
            }
          >
            <option value="">Any installed version</option>
            {pin !== null && (!known || pinNotInstalled) && (
              <option value={String(pin)}>
                {known
                  ? `SolidWorks ${pin} (not installed)`
                  : `SolidWorks ${pin} (saved, not checked)`}
              </option>
            )}
            {known &&
              versions.map((year) => (
                <option key={year} value={String(year)}>
                  {`SolidWorks ${year}`}
                </option>
              ))}
          </select>
          <p id="solidworks-version-help" className="field-help">
            {versionHelp(pin, options.instance_mode)}
          </p>
          {versionMessage && (
            <Message id={versionMessage.id} tone={versionMessage.tone}>
              {versionMessage.text}
            </Message>
          )}
        </div>
        {background ? (
          <NumberField
            inputId="solidworks-max-instances"
            label="Max background instances"
            value={options.max_instances}
            min={1}
            max={MAX_INSTANCES_LIMIT}
            helpId="solidworks-max-instances-help"
            help="How many SolidWorks processes the host may run at once. Each uses memory and may use a licence seat. In testing, two instances under one Windows account used one floating licence; check your own licence terms."
            errorId="solidworks-max-instances-error"
            error={() => `Enter a whole number from 1 to ${MAX_INSTANCES_LIMIT}.`}
            onChange={(next) => update({ max_instances: next })}
          />
        ) : (
          timeoutField
        )}
      </div>
      {background && (
        <div className="form-row">
          {timeoutField}
          <NumberField
            inputId="solidworks-idle-minutes"
            label="Close idle instances after (minutes)"
            value={Math.round(options.idle_shutdown_seconds / 60)}
            min={1}
            max={IDLE_MINUTES_MAX}
            helpId="solidworks-idle-minutes-help"
            help="A background instance exits after this long without a call, which frees memory and licence seats. The next call starts a new one, which takes about 10 seconds."
            errorId="solidworks-idle-minutes-error"
            error={() => `Enter a whole number from 1 to ${IDLE_MINUTES_MAX}.`}
            onChange={(minutes) => update({ idle_shutdown_seconds: minutes * 60 })}
          />
        </div>
      )}
    </fieldset>
  );
}
