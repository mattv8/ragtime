import { useMemo } from 'react';

interface ContextUsagePieProps {
  currentTokens: number;
  totalTokens: number;
  contextLimit: number;
}

export function ContextUsagePie({ currentTokens, totalTokens, contextLimit }: ContextUsagePieProps) {
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

  return (
    <div
      className="context-usage-pie"
      title={`Context: ${percentage}% (${totalTokens.toLocaleString()} / ${contextLimit.toLocaleString()} tokens, current message: ${currentTokens.toLocaleString()})`}
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
          stroke="var(--border-color)"
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
      </svg>
    </div>
  );
}
