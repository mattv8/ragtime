import type React from 'react';
import {
  defaultScheduleStartMinute,
  defaultScheduleTimezone,
  ScheduleStartTimeInput,
} from './ScheduleStartTimeInput';

interface ReindexIntervalSelectProps {
  value: number;
  onChange: (value: number) => void;
  startMinute?: number | null;
  timezone?: string | null;
  onStartMinuteChange?: (value: number | null) => void;
  onTimezoneChange?: (value: string | null) => void;
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
  label?: string;
}

/**
 * Reusable dropdown for selecting auto re-index interval.
 * Used in Git index wizard and filesystem indexer configuration.
 */
export function ReindexIntervalSelect({
  value,
  onChange,
  startMinute,
  timezone,
  onStartMinuteChange,
  onTimezoneChange,
  disabled = false,
  className,
  style,
  label = 'Auto Re-index Interval',
}: ReindexIntervalSelectProps) {
  const handleIntervalChange = (nextValue: number) => {
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

  const showSchedule = value > 0 && onStartMinuteChange && onTimezoneChange;

  return (
    <div className={className} style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', ...style }}>
      <div className="form-group" style={{ flex: '1 1 160px', minWidth: '160px', margin: 0 }}>
        <label>{label}</label>
        <select
          value={value}
          onChange={(e) => handleIntervalChange(parseInt(e.target.value, 10))}
          disabled={disabled}
        >
          <option value={0}>Manual only</option>
          <option value={1}>Every hour</option>
          <option value={6}>Every 6 hours</option>
          <option value={12}>Every 12 hours</option>
          <option value={24}>Every 24 hours (daily)</option>
          <option value={168}>Every week</option>
          <option value={336}>Every 2 weeks</option>
          <option value={720}>Every 30 days</option>
        </select>
      </div>
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
  );
}
