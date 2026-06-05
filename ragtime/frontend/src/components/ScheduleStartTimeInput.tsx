import type React from 'react';

const DEFAULT_START_MINUTE = 9 * 60;

const REPRESENTATIVE_TIMEZONES = [
  'Pacific/Midway',
  'Pacific/Honolulu',
  'America/Anchorage',
  'America/Los_Angeles',
  'America/Denver',
  'America/Chicago',
  'America/New_York',
  'America/Halifax',
  'America/Argentina/Buenos_Aires',
  'Atlantic/South_Georgia',
  'Atlantic/Azores',
  'UTC',
  'Europe/London',
  'Europe/Paris',
  'Europe/Helsinki',
  'Europe/Moscow',
  'Asia/Dubai',
  'Asia/Karachi',
  'Asia/Dhaka',
  'Asia/Bangkok',
  'Asia/Shanghai',
  'Asia/Tokyo',
  'Australia/Sydney',
  'Pacific/Auckland',
];

function getTzShortName(tz: string): string {
  try {
    const parts = new Intl.DateTimeFormat('en-US', { timeZone: tz, timeZoneName: 'short' }).formatToParts();
    return parts.find((p) => p.type === 'timeZoneName')?.value || tz;
  } catch {
    return tz;
  }
}

function getLocalTimezone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
}

function minuteToParts(startMinute: number | null | undefined): { hour: number; minute: number; period: 'AM' | 'PM' } {
  const normalized = typeof startMinute === 'number' && startMinute >= 0 && startMinute < 1440
    ? startMinute
    : DEFAULT_START_MINUTE;
  const hour24 = Math.floor(normalized / 60);
  const minute = normalized % 60;
  const period = hour24 >= 12 ? 'PM' : 'AM';
  const hour = hour24 % 12 || 12;
  return { hour, minute, period };
}

function partsToMinute(hour: number, minute: number, period: 'AM' | 'PM'): number {
  const hour12 = Math.max(1, Math.min(12, hour));
  const normalizedMinute = Math.max(0, Math.min(59, minute));
  const hour24 = period === 'AM'
    ? (hour12 === 12 ? 0 : hour12)
    : (hour12 === 12 ? 12 : hour12 + 12);
  return hour24 * 60 + normalizedMinute;
}

interface ScheduleStartTimeInputProps {
  enabled: boolean;
  startMinute?: number | null;
  timezone?: string | null;
  onStartMinuteChange: (value: number | null) => void;
  onTimezoneChange: (value: string | null) => void;
  disabled?: boolean;
  label?: string;
  className?: string;
  style?: React.CSSProperties;
}

export function defaultScheduleStartMinute(): number {
  return DEFAULT_START_MINUTE;
}

export function defaultScheduleTimezone(): string {
  return getLocalTimezone();
}

export function ScheduleStartTimeInput({
  enabled,
  startMinute,
  timezone,
  onStartMinuteChange,
  onTimezoneChange,
  disabled = false,
  label = 'Start Time',
  className,
  style,
}: ScheduleStartTimeInputProps) {
  if (!enabled) {
    return null;
  }

  const parts = minuteToParts(startMinute);
  const currentTimezone = timezone || getLocalTimezone();
  const allTimezones = Array.from(new Set([...REPRESENTATIVE_TIMEZONES, currentTimezone]));

  const setParts = (next: Partial<typeof parts>) => {
    const merged = { ...parts, ...next };
    onStartMinuteChange(partsToMinute(merged.hour, merged.minute, merged.period));
    if (!timezone) {
      onTimezoneChange(getLocalTimezone());
    }
  };

  return (
    <div className={`form-group schedule-start-time ${className || ''}`} style={style}>
      <label>{label}</label>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'nowrap' }}>
        <select
          value={parts.hour}
          onChange={(event) => setParts({ hour: parseInt(event.target.value, 10) })}
          disabled={disabled}
          aria-label={`${label} hour`}
          style={{ width: '100%', flex: 1, minWidth: 40, paddingRight: 4, paddingLeft: 4 }}
        >
          {Array.from({ length: 12 }, (_, index) => index + 1).map((hour) => (
            <option key={hour} value={hour}>{hour}</option>
          ))}
        </select>
        <select
          value={parts.minute}
          onChange={(event) => setParts({ minute: parseInt(event.target.value, 10) })}
          disabled={disabled}
          aria-label={`${label} minute`}
          style={{ width: '100%', flex: 1, minWidth: 40, paddingRight: 4, paddingLeft: 4 }}
        >
          {[0, 15, 30, 45].map((minute) => (
            <option key={minute} value={minute}>{minute.toString().padStart(2, '0')}</option>
          ))}
        </select>
        <select
          value={parts.period}
          onChange={(event) => setParts({ period: event.target.value as 'AM' | 'PM' })}
          disabled={disabled}
          aria-label={`${label} period`}
          style={{ width: '100%', flex: 1, minWidth: 45, paddingRight: 4, paddingLeft: 4 }}
        >
          <option value="AM">AM</option>
          <option value="PM">PM</option>
        </select>
        <select
          value={currentTimezone}
          onChange={(event) => onTimezoneChange(event.target.value || null)}
          disabled={disabled}
          aria-label={`${label} timezone`}
          style={{ width: '100%', flex: 2, minWidth: 80, textOverflow: 'ellipsis' }}
        >
          {allTimezones.map((tz) => {
            const shortName = getTzShortName(tz);
            const isSelected = tz === currentTimezone;
            return (
              <option key={tz} value={tz}>
                {isSelected ? shortName : `${tz} (${shortName})`}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
}
