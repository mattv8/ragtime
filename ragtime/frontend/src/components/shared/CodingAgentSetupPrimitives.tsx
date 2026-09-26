import { InlineCopyButton } from './InlineCopyButton';
import { useState } from 'react';

export function CopyableSnippet({
  value,
  label = 'Copy configuration',
}: {
  value: string;
  label?: string;
}) {
  return (
    <div className="coding-agent-copyable-snippet" data-setup-snippet={label}>
      <div className="coding-agent-copyable-snippet-toolbar">
        <span className="coding-agent-copyable-snippet-label">{label}</span>
        <InlineCopyButton
          copyText={value}
          className="btn btn-secondary btn-sm"
          title={label}
          ariaLabel={label}
          label="Copy"
        />
      </div>
      <pre>
        <code>{value}</code>
      </pre>
    </div>
  );
}

type ConfigLocation = string | { label: string; path: string };

export function ConfigLocations({ locations }: { locations: ConfigLocation[] }) {
  return (
    <ul className="coding-agent-config-locations" data-setup-config-locations>
      {locations.map((location) => (
        <li
          className="coding-agent-config-location"
          key={typeof location === 'string' ? location : `${location.label}-${location.path}`}
        >
          {typeof location === 'string' ? (
            <code>{location}</code>
          ) : (
            <>
              <span className="coding-agent-config-location-label">{location.label}</span>
              <code>{location.path}</code>
            </>
          )}
        </li>
      ))}
    </ul>
  );
}

export function GuideNote({ children }: { children: React.ReactNode }) {
  return (
    <p className="coding-agent-guide-note" role="note">
      {children}
    </p>
  );
}

export function GuideImage({
  src,
  alt,
  caption,
  href,
}: {
  src: string;
  alt: string;
  caption: string;
  href: string;
}) {
  const [failed, setFailed] = useState(false);
  return (
    <figure className="coding-agent-guide-image" data-guide-image={src}>
      {failed ? (
        <p>
          The example image could not load.{' '}
          <a href={href} target="_blank" rel="noreferrer">
            Open the illustrated source instead.
          </a>
        </p>
      ) : (
        <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} />
      )}
      <figcaption>
        {caption}{' '}
        <a href={href} target="_blank" rel="noreferrer">
          Source and illustrated instructions
        </a>
      </figcaption>
    </figure>
  );
}
