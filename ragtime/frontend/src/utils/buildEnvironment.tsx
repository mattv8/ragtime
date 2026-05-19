const rawBuildEnvironment = (import.meta.env.VITE_RAGTIME_ENVIRONMENT || '').trim();
const buildEnvironment = rawBuildEnvironment || (import.meta.env.DEV ? 'dev' : 'main');
const normalizedBuildEnvironment = buildEnvironment.toLowerCase();
const rawVersion = (import.meta.env.VITE_RAGTIME_VERSION || '').trim();
const versionLabel = rawVersion || '';

export const ragtimeBuildEnvironment = buildEnvironment;
export const isMainEnvironment = normalizedBuildEnvironment === 'main' || normalizedBuildEnvironment === 'production';
export const environmentBadgeLabel = isMainEnvironment ? '' : buildEnvironment;

interface BrandNameProps {
  name: string;
}

export function BrandName({ name }: BrandNameProps) {
  return (
    <span className="brand-name">
      <span>{name}</span>
      {environmentBadgeLabel && <span className="environment-badge">{environmentBadgeLabel}</span>}
      {isMainEnvironment && versionLabel && <span className="version-label">{versionLabel}</span>}
    </span>
  );
}