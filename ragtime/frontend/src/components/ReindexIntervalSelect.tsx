import { useId, useState, type ChangeEvent, type CSSProperties } from 'react';
import type { ReactNode } from 'react';
import {
  defaultScheduleStartMinute,
  defaultScheduleTimezone,
  ScheduleStartTimeInput,
} from './ScheduleStartTimeInput';

export interface IntervalOption {
  value: number;
  label: string;
}

const DEFAULT_INTERVAL_OPTIONS: IntervalOption[] = [
  { value: 0, label: 'Manual only' },
  { value: 1, label: 'Every hour' },
  { value: 6, label: 'Every 6 hours' },
  { value: 12, label: 'Every 12 hours' },
  { value: 24, label: 'Every 24 hours (daily)' },
  { value: 168, label: 'Every week' },
  { value: 336, label: 'Every 2 weeks' },
  { value: 720, label: 'Every 30 days' },
];

export interface ReindexIntervalSelectProps {
  value: number;
  onChange: (value: number) => void;
  intervalOptions?: IntervalOption[];
  webhookDeliveryEnabled?: boolean;
  onWebhookDeliveryChange?: (enabled: boolean) => Promise<boolean>;
  startMinute?: number | null;
  timezone?: string | null;
  onStartMinuteChange?: (value: number | null) => void;
  onTimezoneChange?: (value: string | null) => void;
  disabled?: boolean;
  className?: string;
  style?: CSSProperties;
  label?: string;
  action?: ReactNode;
}

/**
 * Reusable dropdown for selecting auto re-index interval.
 * Used in Git index wizard and filesystem indexer configuration.
 */
export function ReindexIntervalSelect({
  value,
  onChange,
  intervalOptions = DEFAULT_INTERVAL_OPTIONS,
  webhookDeliveryEnabled = false,
  onWebhookDeliveryChange,
  startMinute,
  timezone,
  onStartMinuteChange,
  onTimezoneChange,
  disabled = false,
  className,
  style,
  label = 'Auto Re-index Interval',
  action,
}: ReindexIntervalSelectProps) {
  const selectId = useId();
  const dialogTitleId = useId();
  const [pendingInterval, setPendingInterval] = useState<number | null>(null);
  const [webhookTransitionPending, setWebhookTransitionPending] = useState(false);
  const supportsWebhookDelivery = onWebhookDeliveryChange !== undefined;

  const applyInterval = (nextValue: number) => {
    onChange(nextValue);
    if (nextValue > 0 && onStartMinuteChange && onTimezoneChange && startMinute == null) {
      onStartMinuteChange(defaultScheduleStartMinute());
      onTimezoneChange(timezone || defaultScheduleTimezone());
    }
    if (nextValue <= 0 && onStartMinuteChange && onTimezoneChange) {
      onStartMinuteChange(null);
      onTimezoneChange(null);
    }
  };

  const selectedValue = webhookDeliveryEnabled ? 'webhook' : String(value);
  const showSchedule =
    !webhookDeliveryEnabled && value > 0 && onStartMinuteChange && onTimezoneChange;

  const handleSelectChange = async (event: ChangeEvent<HTMLSelectElement>) => {
    const nextValue = event.target.value;
    if (nextValue === 'webhook') {
      await onWebhookDeliveryChange?.(true);
      return;
    }

    const nextInterval = parseInt(nextValue, 10);
    if (webhookDeliveryEnabled && onWebhookDeliveryChange) {
      setPendingInterval(nextInterval);
      return;
    }
    applyInterval(nextInterval);
  };

  const confirmLeavingWebhook = async () => {
    if (pendingInterval === null) return;
    setWebhookTransitionPending(true);
    try {
      const disabled = await onWebhookDeliveryChange?.(false);
      if (disabled) {
        applyInterval(pendingInterval);
        setPendingInterval(null);
      }
    } catch {
      // Keep webhook delivery selected when disabling fails.
    } finally {
      setWebhookTransitionPending(false);
    }
  };

  return (
    <>
      <div
        className={className}
        style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', ...style }}
      >
        <div className="form-group" style={{ flex: '1 1 160px', minWidth: '160px', margin: 0 }}>
          <label htmlFor={selectId}>{label}</label>
          <select
            id={selectId}
            value={selectedValue}
            onChange={(event) => {
              void handleSelectChange(event);
            }}
            disabled={disabled || webhookTransitionPending}
          >
            {intervalOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
            {supportsWebhookDelivery && <option value="webhook">Webhook delivery</option>}
          </select>
        </div>
        {action && <div className="reindex-interval-action">{action}</div>}
        {showSchedule && (
          <ScheduleStartTimeInput
            enabled={value > 0}
            startMinute={startMinute}
            timezone={timezone}
            onStartMinuteChange={onStartMinuteChange}
            onTimezoneChange={onTimezoneChange}
            disabled={disabled}
            label="Start Time"
            style={{ flex: '2 1 230px', minWidth: '230px', margin: 0 }}
          />
        )}
      </div>
      {pendingInterval !== null && (
        <div className="modal-overlay" onClick={() => setPendingInterval(null)} role="presentation">
          <div
            className="modal-content"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby={dialogTitleId}
          >
            <div className="modal-header">
              <h3 id={dialogTitleId}>Disable webhook delivery?</h3>
            </div>
            <div className="modal-body">
              <p>Switching away from webhook delivery will re-enable scheduled updates.</p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setPendingInterval(null)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={() => {
                  void confirmLeavingWebhook();
                }}
                disabled={webhookTransitionPending}
              >
                Disable webhook and continue
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
