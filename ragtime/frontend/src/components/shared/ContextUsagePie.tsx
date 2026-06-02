import { useMemo } from 'react';

interface ContextUsagePieProps {
  currentTokens: number;
  totalTokens: number;
  contextLimit: number;
  compactThresholdPercent?: number;
  loading?: boolean;
  onCompact?: () => void;
  isCompacting?: boolean;
}

export function ContextUsagePie({ currentTokens, totalTokens, contextLimit, compactThresholdPercent = 80, loading, onCompact, isCompacting }: ContextUsagePieProps) {
  const { percentage, color } = useMemo(() => {
    const pct = contextLimit > 0 ? Math.round((totalTokens / contextLimit) * 100) : 0;

    let colorVar: string;
    if (pct > 90) {
      colorVar = 'var(--color-error)';
    } else if (pct >= 70) {
      colorVar = 'var(--color-warning)';
    } else {
      colorVar = 'var(--color-success)';
    }

    return { percentage: pct, color: colorVar };
  }, [totalTokens, contextLimit]);

  const size = 30;
  const strokeWidth = 3;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const usedArc = (Math.min(percentage, 100) / 100) * circumference;
  const remainingArc = circumference - usedArc;
  const center = size / 2;
  const threshold = Math.max(1, Math.min(100, Math.round(compactThresholdPercent)));
  const canCompact = Boolean(onCompact) && percentage >= threshold;
  const isLoading = loading || isCompacting;
  const title = canCompact
    ? `Compact conversation context (${percentage}% full, threshold: ${threshold}%, ${totalTokens.toLocaleString()} / ${contextLimit.toLocaleString()} tokens)`
    : `Context: ${percentage}% (${totalTokens.toLocaleString()} / ${contextLimit.toLocaleString()} tokens, current message: ${currentTokens.toLocaleString()})`;

  if (isLoading) {
    const loadingArc = circumference * 0.25;
    const loadingGap = circumference - loadingArc;
    const loadingContent = (
      <>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
        >
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="var(--color-border)"
            strokeWidth={strokeWidth}
            opacity="0.2"
          />
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="var(--color-text-muted)"
            strokeWidth={strokeWidth}
            strokeDasharray={`${loadingArc} ${loadingGap}`}
            strokeLinecap="round"
            opacity="0.8"
          >
            <animateTransform
              attributeName="transform"
              type="rotate"
              from={`0 ${center} ${center}`}
              to={`360 ${center} ${center}`}
              dur="1s"
              repeatCount="indefinite"
            />
          </circle>
        </svg>
      </>
    );
    if (canCompact) {
      return (
        <button
          type="button"
          className="context-usage-pie context-usage-pie-button"
          title={isCompacting ? 'Compacting context...' : 'Loading context info...'}
          aria-label={isCompacting ? 'Compacting context' : 'Loading context info'}
          disabled
        >
          {loadingContent}
        </button>
      );
    }
    return (
      <div
        className="context-usage-pie"
        title={isCompacting ? 'Compacting context...' : 'Loading context info...'}
        style={{
          position: 'relative',
          width: `${size}px`,
          height: `${size}px`,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        {loadingContent}
      </div>
    );
  }

  const pieContent = (
    <>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ transform: 'rotate(-90deg)' }}
      >
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth={strokeWidth}
          opacity="0.2"
        />
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={`${usedArc} ${remainingArc}`}
          strokeLinecap="round"
        />
      </svg>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ position: 'absolute', top: 0, left: 0 }}
      >
        <text
          className={canCompact ? 'context-usage-pie-percent-text' : undefined}
          x={center}
          y={center}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize="11"
          fontWeight="700"
          fill={color}
          stroke="var(--color-surface-hover)"
          strokeWidth="3"
          paintOrder="stroke fill"
        >
          {percentage}%
        </text>
        {canCompact && (
          <g className="context-usage-pie-compact-icon" transform={`translate(${center - 8}, ${center - 8}) scale(0.666)`}>
            <g
              stroke="var(--color-surface-hover)"
              strokeWidth="5"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            >
              <path d="m15 15 6 6" />
              <path d="M15 21v-6h6" />
              <path d="m9 9-6-6" />
              <path d="M9 3v6H3" />
            </g>
            <g
              stroke={color}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            >
              <path d="m15 15 6 6" />
              <path d="M15 21v-6h6" />
              <path d="m9 9-6-6" />
              <path d="M9 3v6H3" />
            </g>
          </g>
        )}
      </svg>
    </>
  );

  if (canCompact) {
    return (
      <button
        type="button"
        className="context-usage-pie context-usage-pie-button context-usage-pie-compact-mode"
        title={title}
        aria-label="Compact conversation context"
        onClick={onCompact}
      >
        {pieContent}
      </button>
    );
  }

  return (
    <div
      className="context-usage-pie"
      title={title}
      style={{
        position: 'relative',
        width: `${size}px`,
        height: `${size}px`,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      {pieContent}
    </div>
  );
}
