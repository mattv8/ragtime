export function normalizeShareSlugInput(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[_-]+|[_-]+$/g, '')
    .slice(0, 80);
}

export function getDefaultShareSlug(value: string | null | undefined): string {
  const normalized = normalizeShareSlugInput(value ?? '');
  if (normalized && !normalized.startsWith('workspace')) {
    return `share_${normalized.slice(0, 24)}`;
  }
  return 'share_workspace';
}

export function normalizeUniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort();
}

export function areSameNormalizedStringArrays(left: string[], right: string[]): boolean {
  const normalizedLeft = normalizeUniqueStrings(left);
  const normalizedRight = normalizeUniqueStrings(right);
  if (normalizedLeft.length !== normalizedRight.length) {
    return false;
  }
  return normalizedLeft.every((value, index) => value === normalizedRight[index]);
}