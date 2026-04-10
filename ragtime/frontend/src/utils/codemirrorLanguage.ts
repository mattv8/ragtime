import { useEffect, useState } from 'react';
import type { Extension } from '@codemirror/state';

type LanguageLoader = {
  key: string;
  load: () => Promise<Extension>;
};

const languageExtensionPromises = new Map<string, Promise<Extension | null>>();

function resolveLanguageLoader(filePath: string): LanguageLoader | null {
  const lower = filePath.toLowerCase();

  if (/\.[cm]?tsx$/i.test(lower)) {
    return {
      key: 'tsx',
      load: async () => {
        const { javascript } = await import('@codemirror/lang-javascript');
        return javascript({ typescript: true, jsx: true });
      },
    };
  }

  if (/\.[cm]?ts$/i.test(lower)) {
    return {
      key: 'ts',
      load: async () => {
        const { javascript } = await import('@codemirror/lang-javascript');
        return javascript({ typescript: true, jsx: false });
      },
    };
  }

  if (/\.[cm]?jsx$/i.test(lower)) {
    return {
      key: 'jsx',
      load: async () => {
        const { javascript } = await import('@codemirror/lang-javascript');
        return javascript({ typescript: false, jsx: true });
      },
    };
  }

  if (/\.[cm]?js$/i.test(lower)) {
    return {
      key: 'js',
      load: async () => {
        const { javascript } = await import('@codemirror/lang-javascript');
        return javascript({ typescript: false, jsx: false });
      },
    };
  }

  if (/\.py$/i.test(lower)) {
    return {
      key: 'py',
      load: async () => {
        const { python } = await import('@codemirror/lang-python');
        return python();
      },
    };
  }

  if (/\.json[c5]?$/i.test(lower) || lower.endsWith('.jsonl')) {
    return {
      key: 'json',
      load: async () => {
        const { json } = await import('@codemirror/lang-json');
        return json();
      },
    };
  }

  if (/\.css$/i.test(lower) || lower.endsWith('.scss') || lower.endsWith('.less')) {
    return {
      key: 'css',
      load: async () => {
        const { css } = await import('@codemirror/lang-css');
        return css();
      },
    };
  }

  if (/\.html?$/i.test(lower) || lower.endsWith('.svelte') || lower.endsWith('.vue')) {
    return {
      key: 'html',
      load: async () => {
        const { html } = await import('@codemirror/lang-html');
        return html();
      },
    };
  }

  if (/\.ya?ml$/i.test(lower)) {
    return {
      key: 'yaml',
      load: async () => {
        const { yaml } = await import('@codemirror/lang-yaml');
        return yaml();
      },
    };
  }

  if (/\.xml$/i.test(lower) || lower.endsWith('.svg')) {
    return {
      key: 'xml',
      load: async () => {
        const { xml } = await import('@codemirror/lang-xml');
        return xml();
      },
    };
  }

  if (/\.sql$/i.test(lower)) {
    return {
      key: 'sql',
      load: async () => {
        const { sql } = await import('@codemirror/lang-sql');
        return sql();
      },
    };
  }

  if (/\.mdx?$/i.test(lower) || lower.endsWith('.markdown')) {
    return {
      key: 'markdown',
      load: async () => {
        const { markdown } = await import('@codemirror/lang-markdown');
        return markdown();
      },
    };
  }

  return null;
}

export async function loadCodeMirrorLanguageExtension(
  filePath: string,
): Promise<Extension | null> {
  const loader = resolveLanguageLoader(filePath);
  if (!loader) {
    return null;
  }

  let pending = languageExtensionPromises.get(loader.key);
  if (!pending) {
    pending = loader
      .load()
      .then((extension) => extension)
      .catch((error) => {
        languageExtensionPromises.delete(loader.key);
        throw error;
      });
    languageExtensionPromises.set(loader.key, pending);
  }

  return pending;
}

export function useCodeMirrorLanguageExtension(
  filePath: string,
): Extension | null {
  const [extension, setExtension] = useState<Extension | null>(null);

  useEffect(() => {
    let cancelled = false;

    if (!filePath) {
      setExtension(null);
      return () => {
        cancelled = true;
      };
    }

    setExtension(null);
    void loadCodeMirrorLanguageExtension(filePath)
      .then((nextExtension) => {
        if (!cancelled) {
          setExtension(nextExtension);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setExtension(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [filePath]);

  return extension;
}