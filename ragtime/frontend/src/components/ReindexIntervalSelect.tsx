interface ReindexIntervalSelectProps {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Reusable dropdown for selecting auto re-index interval.
 * Used in Git index wizard and filesystem indexer configuration.
 */
export function ReindexIntervalSelect({
  value,
  onChange,
  disabled = false,
  className,
  style,
}: ReindexIntervalSelectProps) {
  return (
    <div className={`form-group ${className || ''}`} style={style}>
      <label>Auto Re-index Interval</label>
      <select
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        disabled={disabled}
      >
        <option value={0}>Manual only</option>
        <option value={1}>Every hour</option>
        <option value={6}>Every 6 hours</option>
        <option value={12}>Every 12 hours</option>
        <option value={24}>Every 24 hours (daily)</option>
        <option value={168}>Every week</option>
      </select>
    </div>
  );
}
