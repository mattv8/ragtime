import { InlineCopyButton } from './InlineCopyButton';

interface RecoveryCodesDisplayProps {
  codes: string[];
}

/**
 * Renders MFA recovery codes with a button to copy all codes at once.
 * Shared across TOTP and passkey (WebAuthn) enrollment surfaces.
 */
export function RecoveryCodesDisplay({ codes }: RecoveryCodesDisplayProps) {
  return (
    <div className="recovery-codes">
      <div className="cloud-oauth-callback-code recovery-codes-list">
        {codes.map((code) => (
          <div key={code}>{code}</div>
        ))}
      </div>
      <InlineCopyButton
        copyText={() => codes.join('\n')}
        className="btn btn-secondary btn-sm recovery-codes-copy"
        label="Copy codes"
        copiedLabel="Copied"
        title="Copy all recovery codes"
        ariaLabel="Copy all recovery codes"
        copiedTitle="Recovery codes copied"
        copiedAriaLabel="Recovery codes copied"
        iconSize={14}
      />
    </div>
  );
}
