interface OAuthCallbackErrorProps {
  title: string;
  summary: string;
  nextSteps?: string[];
}

export function OAuthCallbackError({ title, summary, nextSteps = [] }: OAuthCallbackErrorProps) {
  return (
    <div className="login-container">
      <div className="login-card" style={{ maxWidth: '560px' }}>
        <p
          style={{
            margin: '0 0 8px 0',
            color: 'var(--color-error)',
            fontSize: 'var(--text-xs)',
            fontWeight: 700,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}
        >
          OAuth Callback Error
        </p>
        <h1
          style={{
            fontSize: 'var(--text-2xl)',
            lineHeight: 1.2,
            margin: '0 0 12px 0',
            color: 'var(--color-text)',
          }}
        >
          {title}
        </h1>
        <p
          style={{
            margin: 0,
            color: 'var(--color-text-secondary)',
            fontSize: 'var(--text-sm)',
            lineHeight: 1.6,
          }}
        >
          {summary}
        </p>
        {nextSteps.length > 0 && (
          <div style={{ marginTop: '14px' }}>
            <p
              style={{
                margin: '0 0 8px 0',
                color: 'var(--color-text)',
                fontSize: 'var(--text-xs)',
                fontWeight: 600,
              }}
            >
              Next steps
            </p>
            <ol
              style={{
                margin: 0,
                paddingLeft: '18px',
                color: 'var(--color-text-secondary)',
                fontSize: 'var(--text-sm)',
              }}
            >
              {nextSteps.map((step, i) => (
                <li key={i} style={{ marginBottom: '8px' }}>
                  {step}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}
