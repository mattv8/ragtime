import { useEffect, useState, type ReactNode } from 'react';
import { AlertTriangle, Info, XCircle, type LucideIcon } from 'lucide-react';
import {
  SOLIDWORKS_DEFAULT_OPTIONS,
  type SolidworksHostStatus,
  type SolidworksInstanceMode,
  type SolidworksOptions,
} from '../api/solidworks';
import { InfoPopover } from './shared/InfoPopover';

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
const TIMEOUT_PROXY_WARNING_ABOVE = 110;
const IDLE_MINUTES_MAX = 1440;
const MAX_INSTANCES_LIMIT = 4;

const INSTANCE_MODES: ReadonlyArray<readonly [SolidworksInstanceMode, string, string]> = [
  [
    'background',
    'Background instance',
    'The agent starts its own hidden SolidWorks and closes it when idle. Best for a shared or unattended machine.',
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
  infoId: string;
  infoLabel: string;
  info: ReactNode;
  unit?: 'sec' | 'min';
  errorId: string;
  /** Receives the parsed draft (NaN when empty or not a number). */
  error: (draft: number) => string;
  warningId?: string;
  onChange: (value: number) => void;
}

function NumberField({
  inputId,
  label,
  value,
  min,
  max,
  infoId,
  infoLabel,
  info,
  unit,
  errorId,
  error,
  warningId,
  onChange,
}: NumberFieldProps) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const parsed = parseDraft(draft);
  const valid = isWholeInRange(parsed, min, max);
  const describedBy = [!valid && errorId, warningId].filter(Boolean).join(' ') || undefined;

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
      <div className="sw-field-label">
        <label htmlFor={inputId}>
          {label}
          {unit && (
            <span className="sw-visually-hidden"> ({unit === 'sec' ? 'seconds' : 'minutes'})</span>
          )}
        </label>
        <InfoPopover id={infoId} label={infoLabel}>
          {info}
        </InfoPopover>
      </div>
      <div className={unit ? 'sw-unit-input' : undefined}>
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
        {unit && (
          <span className="sw-unit" aria-hidden="true">
            {unit}
          </span>
        )}
      </div>
      {!valid && (
        <Message id={errorId} tone="error">
          {error(parsed)}
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
    ? `The agent only starts SolidWorks ${pin}. Calls fail if it is not installed.`
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
        text: 'Start the agent to choose from installed versions.',
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
  const versionDescribedBy = versionMessage?.id;
  const background = options.instance_mode === 'background';
  const timeoutWarning =
    timeoutMaxSeconds > TIMEOUT_PROXY_WARNING_ABOVE
      ? 'Calls over about 120 seconds may be cut off by the web proxy.'
      : undefined;
  const timeoutField = (
    <NumberField
      inputId="solidworks-timeout-max-seconds"
      label="Timeout"
      unit="sec"
      value={timeoutMaxSeconds}
      min={TIMEOUT_MIN}
      max={TIMEOUT_MAX}
      infoId="solidworks-timeout-help"
      infoLabel="About timeout"
      info="Longest a single SolidWorks call may take, including starting SolidWorks (about 10 seconds) and opening large assemblies. The default is 100 because the web proxy in front of Ragtime ends requests at about 120 seconds."
      errorId="solidworks-timeout-error"
      error={timeoutError}
      warningId={timeoutWarning ? 'solidworks-timeout-warning' : undefined}
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
      <div id="solidworks-instance-mode-field">
        <div className="sw-field-label">
          <span id="solidworks-instance-mode-label">Instance mode</span>
          <InfoPopover id="solidworks-instance-mode-help" label="About instance mode">
            {INSTANCE_MODES.map(([mode, title, description]) => (
              <p key={mode}>
                <strong>{title}:</strong>{' '}
                {description.charAt(0).toLowerCase() + description.slice(1)}
              </p>
            ))}
          </InfoPopover>
        </div>
        <div
          id="solidworks-instance-mode"
          role="radiogroup"
          aria-labelledby="solidworks-instance-mode-label"
          className="sw-segmented"
        >
          {INSTANCE_MODES.map(([mode, title]) => (
            <label key={mode}>
              <input
                id={`solidworks-instance-mode-${mode}`}
                type="radio"
                name="solidworks-instance-mode"
                value={mode}
                checked={options.instance_mode === mode}
                onChange={() => update({ instance_mode: mode })}
              />
              <span>{title}</span>
            </label>
          ))}
        </div>
      </div>
      <div id="solidworks-options-fields" className="sw-options-grid">
        <div className="form-group">
          <div className="sw-field-label">
            <label htmlFor="solidworks-version">Version</label>
            <InfoPopover id="solidworks-version-help" label="About version">
              {versionHelp(pin, options.instance_mode)}
            </InfoPopover>
          </div>
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
        </div>
        {background && (
          <NumberField
            inputId="solidworks-max-instances"
            label="Max instances"
            value={options.max_instances}
            min={1}
            max={MAX_INSTANCES_LIMIT}
            infoId="solidworks-max-instances-help"
            infoLabel="About max instances"
            info="How many SolidWorks processes the agent may run at once. Each uses memory and may use a licence seat. In testing, two instances under one Windows account used one floating licence; check your own licence terms."
            errorId="solidworks-max-instances-error"
            error={() => `Enter a whole number from 1 to ${MAX_INSTANCES_LIMIT}.`}
            onChange={(next) => update({ max_instances: next })}
          />
        )}
        {timeoutField}
        {background && (
          <NumberField
            inputId="solidworks-idle-minutes"
            label="Close idle after"
            unit="min"
            value={Math.round(options.idle_shutdown_seconds / 60)}
            min={1}
            max={IDLE_MINUTES_MAX}
            infoId="solidworks-idle-minutes-help"
            infoLabel="About idle shutdown"
            info="A background instance exits after this long without a call, which frees memory and licence seats. The next call starts a new one, which takes about 10 seconds."
            errorId="solidworks-idle-minutes-error"
            error={() => `Enter a whole number from 1 to ${IDLE_MINUTES_MAX}.`}
            onChange={(minutes) => update({ idle_shutdown_seconds: minutes * 60 })}
          />
        )}
      </div>
      {versionMessage && (
        <Message id={versionMessage.id} tone={versionMessage.tone}>
          {versionMessage.text}
        </Message>
      )}
      {timeoutWarning && (
        <Message id="solidworks-timeout-warning" tone="warning">
          {timeoutWarning}
        </Message>
      )}
      <div id="solidworks-auto-update-row" className="sw-switch-row">
        <label>
          <input
            id="solidworks-auto-update"
            type="checkbox"
            role="switch"
            className="sw-switch"
            checked={options.auto_update_host !== false}
            aria-describedby="solidworks-auto-update-summary"
            onChange={(event) => update({ auto_update_host: event.target.checked })}
          />
          Keep the agent up to date
        </label>
        <InfoPopover id="solidworks-auto-update-help" label="About agent updates">
          <p>
            When on, Ragtime updates the SolidWorks agent by itself whenever this Ragtime server has
            a newer agent version than the one installed.
          </p>
          <p>
            It only does this when the agent is idle: no request is running, none ran in the last 5
            minutes, and no document is open in a SolidWorks instance the agent started.
          </p>
          <p>
            An update restarts the agent and closes the SolidWorks instances it started. A
            SolidWorks you started yourself is not touched. If an update fails, the previous version
            is restored and Ragtime tries again later.
          </p>
          <p>
            When off, use <strong>Update agent</strong> in the agent panel. Saved with the tool.
          </p>
        </InfoPopover>
        <span id="solidworks-auto-update-summary" className="sw-switch-summary">
          {options.auto_update_host !== false
            ? 'On: Ragtime updates the agent when it is idle'
            : 'Off: update the agent yourself from the agent panel'}
        </span>
      </div>
    </fieldset>
  );
}
