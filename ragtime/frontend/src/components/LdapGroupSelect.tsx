interface LdapGroup {
  dn: string;
  name: string;
}

interface LdapGroupSelectProps {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  groups: LdapGroup[];
  emptyOptionLabel?: string;
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
  disabled = false,
  required = false,
  className,
  style,
}: LdapGroupSelectProps) {
  const uniqueGroups = groups.filter(
    (group, index, all) =>
      group?.dn && all.findIndex((candidate) => candidate.dn === group.dn) === index,
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
          {group.name}
        </option>
      ))}
    </select>
  );
}
