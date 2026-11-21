import type { SolidworksInstanceMode } from '../api/solidworks';
import { useContentProtectionEnabled } from '../hooks/useContentProtectionEnabled';
import { PdmMessage } from './shared/PdmMessage';
import { InfoPopover } from './shared/InfoPopover';

const WARNING = 'Content protection is off, so scripts run without review.';
const UNKNOWN =
  'Ragtime could not check whether content protection is on. If it is off, scripts run without review.';

export function SolidWorksScriptsPanel({
  value,
  onChange,
  instanceMode,
  disabled,
}: {
  value: boolean;
  onChange: (next: boolean) => void;
  instanceMode: SolidworksInstanceMode;
  disabled?: boolean;
}) {
  const protection = useContentProtectionEnabled(value);
  const attach = value && instanceMode === 'attach';
  const warning = value && protection === 'off';
  const unknown = value && protection === 'unknown';
  const describedBy = [
    'solidworks-allow-scripts-summary',
    attach && 'solidworks-allow-scripts-attach',
    warning && 'solidworks-allow-scripts-warning',
    unknown && 'solidworks-allow-scripts-unknown',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <fieldset
      id="solidworks-scripts-panel"
      data-testid="solidworks-scripts-panel"
      className="sw-options"
      disabled={disabled}
    >
      <legend>Scripts</legend>
      <div id="solidworks-allow-scripts-row" className="sw-switch-row">
        <label>
          <input
            id="solidworks-allow-scripts"
            type="checkbox"
            role="switch"
            className="sw-switch"
            checked={value}
            onChange={(event) => onChange(event.target.checked)}
            aria-describedby={describedBy}
          />
          Allow scripts
        </label>
        <InfoPopover id="solidworks-allow-scripts-help" label="About scripts">
          <p>When on, the AI can write and run C# scripts that drive SolidWorks.</p>
          <p>
            Scripts run as the Windows user of this computer and are not sandboxed. They can read
            and change any file that user can, use the network, and use PDM with that user&apos;s
            rights.
          </p>
          <p>
            Content protection reviews each script before it runs when its coverage rules include
            this tool.
          </p>
          <p>In attach mode, scripts act on the SolidWorks session you have open.</p>
        </InfoPopover>
        <span id="solidworks-allow-scripts-summary" className="sw-switch-summary">
          {value
            ? "On: scripts run as this computer's Windows user, not sandboxed"
            : 'Off: the AI cannot run scripts'}
        </span>
      </div>
      <div id="solidworks-allow-scripts-status" role="status" aria-live="polite">
        {attach && (
          <PdmMessage id="solidworks-allow-scripts-attach" tone="info">
            Attach mode: scripts work in your open SolidWorks session and can change the documents
            you have open.
          </PdmMessage>
        )}
        {warning && (
          <PdmMessage id="solidworks-allow-scripts-warning" tone="warning" boxed>
            {WARNING}
          </PdmMessage>
        )}
        {unknown && (
          <PdmMessage id="solidworks-allow-scripts-unknown" tone="info">
            {UNKNOWN}
          </PdmMessage>
        )}
      </div>
    </fieldset>
  );
}

export function SolidWorksScriptsReview({ allowed }: { allowed: boolean }) {
  const protection = useContentProtectionEnabled(allowed);
  return (
    <li id="tool-review-solidworks-scripts">
      {allowed
        ? "Scripts: Allowed. They run as the computer's Windows user and are not sandboxed."
        : 'Scripts: Off'}
      {allowed && protection === 'off' && (
        <PdmMessage tone="warning" boxed>
          {WARNING}
        </PdmMessage>
      )}
      {allowed && protection === 'unknown' && <PdmMessage tone="info">{UNKNOWN}</PdmMessage>}
    </li>
  );
}
