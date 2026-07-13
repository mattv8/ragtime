import { X } from 'lucide-react';

export interface LdapGroup {
  dn: string;
  name: string;
  display_name?: string | null;
  displayName?: string | null;
}

export function getLdapGroupDisplayName(group: LdapGroup | undefined, fallbackDn: string): string {
  return (
    group?.display_name?.trim() || group?.displayName?.trim() || group?.name?.trim() || fallbackDn
  );
}

interface LdapGroupSelectProps {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  groups: LdapGroup[];
  emptyOptionLabel?: string;
  excludedDns?: string[];
  disabled?: boolean;
  required?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export function LdapGroupSelect({
  id,
  value,
  onChange,
  groups,
  emptyOptionLabel = 'Select an LDAP group...',
  excludedDns = [],
  disabled = false,
  required = false,
  className,
  style,
}: LdapGroupSelectProps) {
  const excludedDnSet = new Set(excludedDns);
  const uniqueGroups = groups.filter(
    (group, index, all) =>
      group?.dn &&
      !excludedDnSet.has(group.dn) &&
      all.findIndex((candidate) => candidate.dn === group.dn) === index,
  );

  return (
    <select
      id={id}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      disabled={disabled}
      required={required}
      className={className}
      style={style}
    >
      <option value="">{emptyOptionLabel}</option>
      {uniqueGroups.map((group) => (
        <option key={group.dn} value={group.dn}>
          {getLdapGroupDisplayName(group, group.dn)}
        </option>
      ))}
    </select>
  );
}

interface LdapGroupChipsProps {
  selectedDns: string[];
  groups: LdapGroup[];
  onRemove: (dn: string) => void;
  disabled?: boolean;
  emptyLabel?: string;
}

export function LdapGroupChips({
  selectedDns,
  groups,
  onRemove,
  disabled = false,
  emptyLabel,
}: LdapGroupChipsProps) {
  const uniqueDns = selectedDns.filter(
    (dn, index, all) => dn && all.findIndex((candidate) => candidate === dn) === index,
  );

  if (uniqueDns.length === 0) {
    return emptyLabel ? <p className="ldap-group-chips-empty">{emptyLabel}</p> : null;
  }

  return (
    <div className="ldap-group-chips" aria-label="Selected LDAP groups">
      {uniqueDns.map((dn) => {
        const group = groups.find((candidate) => candidate.dn === dn);
        const label = getLdapGroupDisplayName(group, dn);
        return (
          <span key={dn} className="ldap-group-chip" title={dn}>
            <span className="ldap-group-chip-label">{label}</span>
            <button
              type="button"
              className="ldap-group-chip-remove"
              onClick={() => onRemove(dn)}
              disabled={disabled}
              aria-label={`Remove ${label}`}
              title={`Remove ${label}`}
            >
              <X size={12} />
            </button>
          </span>
        );
      })}
    </div>
  );
}
