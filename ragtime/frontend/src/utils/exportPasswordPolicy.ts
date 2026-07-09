export interface ExportPasswordPolicy {
  export_password_min_length: number;
  export_password_require_uppercase: boolean;
  export_password_require_lowercase: boolean;
  export_password_require_number: boolean;
  export_password_require_special: boolean;
}

export const DEFAULT_EXPORT_PASSWORD_POLICY: ExportPasswordPolicy = {
  export_password_min_length: 12,
  export_password_require_uppercase: true,
  export_password_require_lowercase: true,
  export_password_require_number: true,
  export_password_require_special: true,
};

export interface PasswordRequirementResult {
  key: string;
  /** Full, descriptive requirement text (used for accessibility/tooltips). */
  label: string;
  /** Compact label suited to inline chips. */
  shortLabel: string;
  met: boolean;
}

const ONE_UPPERCASE = /[A-Z]/;
const ONE_LOWERCASE = /[a-z]/;
const ONE_NUMBER = /[0-9]/;
const ONE_SPECIAL = /[^A-Za-z0-9]/;

export function getExportPasswordPolicy(
  settings?: Partial<ExportPasswordPolicy> | null,
): ExportPasswordPolicy {
  return {
    export_password_min_length:
      typeof settings?.export_password_min_length === 'number' &&
      settings.export_password_min_length > 0
        ? settings.export_password_min_length
        : DEFAULT_EXPORT_PASSWORD_POLICY.export_password_min_length,
    export_password_require_uppercase:
      typeof settings?.export_password_require_uppercase === 'boolean'
        ? settings.export_password_require_uppercase
        : DEFAULT_EXPORT_PASSWORD_POLICY.export_password_require_uppercase,
    export_password_require_lowercase:
      typeof settings?.export_password_require_lowercase === 'boolean'
        ? settings.export_password_require_lowercase
        : DEFAULT_EXPORT_PASSWORD_POLICY.export_password_require_lowercase,
    export_password_require_number:
      typeof settings?.export_password_require_number === 'boolean'
        ? settings.export_password_require_number
        : DEFAULT_EXPORT_PASSWORD_POLICY.export_password_require_number,
    export_password_require_special:
      typeof settings?.export_password_require_special === 'boolean'
        ? settings.export_password_require_special
        : DEFAULT_EXPORT_PASSWORD_POLICY.export_password_require_special,
  };
}

export function getPasswordRequirementResults(
  password: string,
  policy: ExportPasswordPolicy,
): PasswordRequirementResult[] {
  const allResults: PasswordRequirementResult[] = [
    {
      key: 'min_length',
      label: `At least ${policy.export_password_min_length} characters`,
      shortLabel: `${policy.export_password_min_length}+ characters`,
      met: password.length >= policy.export_password_min_length,
    },
    {
      key: 'uppercase',
      label: 'One uppercase letter',
      shortLabel: 'Uppercase',
      met: ONE_UPPERCASE.test(password),
    },
    {
      key: 'lowercase',
      label: 'One lowercase letter',
      shortLabel: 'Lowercase',
      met: ONE_LOWERCASE.test(password),
    },
    {
      key: 'number',
      label: 'One number',
      shortLabel: 'Number',
      met: ONE_NUMBER.test(password),
    },
    {
      key: 'special',
      label: 'One special character',
      shortLabel: 'Special character',
      met: ONE_SPECIAL.test(password),
    },
  ];

  return allResults.filter((result) => {
    switch (result.key) {
      case 'min_length':
        return true;
      case 'uppercase':
        return policy.export_password_require_uppercase;
      case 'lowercase':
        return policy.export_password_require_lowercase;
      case 'number':
        return policy.export_password_require_number;
      case 'special':
        return policy.export_password_require_special;
      default:
        return false;
    }
  });
}

export function passwordMeetsRequirements(password: string, policy: ExportPasswordPolicy): boolean {
  return getPasswordRequirementResults(password, policy).every((result) => result.met);
}
