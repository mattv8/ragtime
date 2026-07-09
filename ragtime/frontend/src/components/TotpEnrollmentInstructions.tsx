import { QRCodeSVG } from 'qrcode.react';
import { Info } from 'lucide-react';

import { Popover } from './Popover';
import { InlineCopyButton } from './shared/InlineCopyButton';

interface TotpEnrollmentInstructionsProps {
  secret: string;
  otpauthUri: string;
  compact?: boolean;
}

const TOTP_HELP_CONTENT = (
  <div className="totp-help-popover">
    <p>
      Open a TOTP authenticator app such as Duo Mobile, 1Password, Google Authenticator, or
      Microsoft Authenticator.
    </p>
    <ol>
      <li>Choose the option to add an account or scan a QR code.</li>
      <li>Scan this code, or enter the manual setup key below.</li>
      <li>Enter the 6-digit code from the app to finish setup.</li>
    </ol>
  </div>
);

export function TotpQrCard({
  otpauthUri,
  compact = false,
}: {
  otpauthUri: string;
  compact?: boolean;
}) {
  return (
    <div
      className="totp-qr-frame totp-qr-dot-matrix"
      aria-label="Authenticator app QR code"
      role="img"
    >
      <QRCodeSVG
        value={otpauthUri}
        size={compact ? 132 : 168}
        marginSize={3}
        level="Q"
        bgColor="transparent"
        fgColor="#141413"
        aria-hidden="true"
        focusable={false}
      />
    </div>
  );
}

export function TotpInstructions() {
  return (
    <div className="totp-enrollment-copy">
      <div className="totp-enrollment-title-row">
        <h3 className="totp-enrollment-title">Scan this QR code</h3>
        <Popover
          content={TOTP_HELP_CONTENT}
          position="right"
          trigger="hover"
          className="totp-help-wrap"
        >
          <button
            type="button"
            className="totp-help-trigger totp-help-trigger-borderless"
            aria-label="Authenticator setup help"
          >
            <Info size={15} aria-hidden="true" />
          </button>
        </Popover>
      </div>
      <p className="totp-enrollment-help">
        Scan with Duo Mobile, 1Password, Google Authenticator, or Microsoft Authenticator. Then
        enter the 6-digit code from the app to finish setup.
      </p>
    </div>
  );
}

export function TotpManualSetup({ secret, otpauthUri }: { secret: string; otpauthUri: string }) {
  return (
    <details className="totp-manual-setup">
      <summary>Manual setup</summary>
      <div className="form-group">
        <label className="form-label">Manual setup key</label>
        <div className="totp-code-row">
          <code className="cloud-oauth-callback-code totp-code-block">{secret}</code>
          <InlineCopyButton
            copyText={secret}
            className="totp-inline-copy"
            title="Copy manual setup key"
            ariaLabel="Copy manual setup key"
            copiedTitle="Key copied"
            copiedAriaLabel="Key copied"
            iconSize={14}
          />
        </div>
      </div>
      <div className="form-group">
        <label className="form-label">Authenticator URI</label>
        <div className="totp-code-row">
          <code className="cloud-oauth-callback-code totp-code-block">{otpauthUri}</code>
          <InlineCopyButton
            copyText={otpauthUri}
            className="totp-inline-copy"
            title="Copy authenticator URI"
            ariaLabel="Copy authenticator URI"
            copiedTitle="URI copied"
            copiedAriaLabel="URI copied"
            iconSize={14}
          />
        </div>
      </div>
    </details>
  );
}

export function TotpEnrollmentInstructions({
  secret,
  otpauthUri,
  compact = false,
}: TotpEnrollmentInstructionsProps) {
  return (
    <div className={`totp-enrollment${compact ? ' totp-enrollment-compact' : ''}`}>
      <div className="totp-enrollment-card">
        <TotpQrCard otpauthUri={otpauthUri} compact={compact} />
        <TotpInstructions />
      </div>
      <TotpManualSetup secret={secret} otpauthUri={otpauthUri} />
    </div>
  );
}
