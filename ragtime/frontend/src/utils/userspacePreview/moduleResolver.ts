export const SUPPORTED_EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx'] as const;

export type ModuleMap = Record<string, string>;

export function isLocalSpecifier(specifier: string): boolean {
  return specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/');
}

export function normalizePath(input: string): string {
  const trimmed = (input || '').trim();
  if (!trimmed) return '';

  const withoutBackslashes = trimmed.replace(/\\/g, '/');
  const withoutLeadingSlash = withoutBackslashes.replace(/^\//, '');
  const segments = withoutLeadingSlash.split('/');
  const normalized: string[] = [];

  for (const segment of segments) {
    if (!segment || segment === '.') continue;
    if (segment === '..') {
      if (normalized.length > 0) {
        normalized.pop();
      }
      continue;
    }
    normalized.push(segment);
  }

  return normalized.join('/');
}

export function dirname(path: string): string {
  const normalized = normalizePath(path);
  const index = normalized.lastIndexOf('/');
  return index === -1 ? '' : normalized.slice(0, index);
}

function hasSupportedExtension(path: string): boolean {
  return SUPPORTED_EXTENSIONS.some((ext) => path.endsWith(ext));
}

function resolveImportTarget(importerPath: string, specifier: string): string {
  const trimmedSpecifier = (specifier || '').trim();
  if (!trimmedSpecifier) return '';

  if (trimmedSpecifier.startsWith('/')) {
    return normalizePath(trimmedSpecifier);
  }

  const importerDir = dirname(importerPath);
  return normalizePath(`${importerDir}/${trimmedSpecifier}`);
}

export function resolveWorkspaceModulePath(
  importerPath: string,
  specifier: string,
  fileMap: ModuleMap
): string | null {
  if (!isLocalSpecifier(specifier)) {
    return null;
  }

  const basePath = resolveImportTarget(importerPath, specifier);
  if (!basePath) return null;

  const candidates: string[] = [basePath];
  if (!hasSupportedExtension(basePath)) {
    for (const extension of SUPPORTED_EXTENSIONS) {
      candidates.push(`${basePath}${extension}`);
    }
    for (const extension of SUPPORTED_EXTENSIONS) {
      candidates.push(`${basePath}/index${extension}`);
    }
  } else if (basePath.endsWith('.js') || basePath.endsWith('.jsx')) {
    const withoutExtension = basePath.replace(/\.(js|jsx)$/i, '');
    candidates.push(`${withoutExtension}.ts`);
    candidates.push(`${withoutExtension}.tsx`);
  } else if (basePath.endsWith('.ts') || basePath.endsWith('.tsx')) {
    const withoutExtension = basePath.replace(/\.(ts|tsx)$/i, '');
    candidates.push(`${withoutExtension}.js`);
    candidates.push(`${withoutExtension}.jsx`);
  }

  for (const candidate of candidates) {
    if (Object.prototype.hasOwnProperty.call(fileMap, candidate)) {
      return candidate;
    }
  }

  return null;
}

export function collectLocalSpecifiers(source: string): string[] {
  const values: string[] = [];
  const staticImportPattern = /(?:import|export)\s+(?:[^'"`]*?\s*from\s*)?['"]([^'"]+)['"]/g;
  const dynamicImportPattern = /import\(\s*['"]([^'"]+)['"]\s*\)/g;

  for (const pattern of [staticImportPattern, dynamicImportPattern]) {
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(source)) !== null) {
      const specifier = match[1];
      if (specifier) values.push(specifier);
    }
  }

  return values;
}

export function rewriteLocalSpecifiers(
  source: string,
  importerPath: string,
  moduleUrlMap: Record<string, string>,
  availableModules: ModuleMap
): { output: string; errors: string[] } {
  const errors: string[] = [];

  const rewrite = (specifier: string): string => {
    const resolved = resolveWorkspaceModulePath(importerPath, specifier, availableModules);
    if (!resolved) {
      if (isLocalSpecifier(specifier)) {
        errors.push(`${importerPath}: unresolved local import '${specifier}'.`);
      }
      return specifier;
    }
    const rewritten = moduleUrlMap[resolved];
    if (!rewritten) {
      errors.push(`${importerPath}: resolved import '${specifier}' to '${resolved}' but module URL is missing.`);
      return specifier;
    }
    return rewritten;
  };

  const patterns: RegExp[] = [
    /(\bimport\s*['"])([^'"]+)(['"])/g,
    /(\bfrom\s*['"])([^'"]+)(['"])/g,
    /(\bimport\(\s*['"])([^'"]+)(['"]\s*\))/g,
  ];

  let output = source;
  for (const pattern of patterns) {
    output = output.replace(pattern, (_full, prefix: string, specifier: string, suffix: string) => {
      return `${prefix}${rewrite(specifier)}${suffix}`;
    });
  }

  return { output, errors };
}
