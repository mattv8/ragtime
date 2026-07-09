import {
  getPasswordRequirementResults,
  type ExportPasswordPolicy,
} from '@/utils/exportPasswordPolicy';

interface PasswordRequirementsChecklistProps {
  password: string;
  policy: ExportPasswordPolicy;
}

/**
 * Compact, negative-space password requirement hint.
 *
 * Renders only the requirements the current password has NOT met yet as inline
 * chips that wrap. Once every requirement is satisfied it collapses to a single
 * success line, keeping the modal quiet when the password is already valid.
 */
export function PasswordRequirementsChecklist({
  password,
  policy,
}: PasswordRequirementsChecklistProps) {
  const requirements = getPasswordRequirementResults(password, policy);
  const unmet = requirements.filter((requirement) => !requirement.met);

  if (unmet.length === 0) {
    return (
      <p className="password-requirements-success" role="status">
        <span className="password-requirements-success-icon" aria-hidden="true">
          ✓
        </span>
        Password meets all requirements
      </p>
    );
  }

  return (
    <div className="password-requirements" aria-label="Password requirements not yet met">
      <ul className="password-requirements-chips">
        {unmet.map((requirement) => (
          <li key={requirement.key} className="password-requirement-chip" title={requirement.label}>
            {requirement.shortLabel}
          </li>
        ))}
      </ul>
    </div>
  );
}
