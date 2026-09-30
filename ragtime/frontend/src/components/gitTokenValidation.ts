import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';

export type GitTokenCheckState = 'idle' | 'checking' | 'valid' | 'invalid' | 'failed';

export function isGitAuthenticationError(message: string): boolean {
  if (/rate[\s-]*limit|too many requests|\b429\b/i.test(message)) return false;
  return /(auth(?:entication)?\s+(?:required|failed)|invalid credentials|invalid username or token|bad credentials|could not read (?:username|password)|username\/password|access denied|repository not found|repository could not be found or read|http\s*(?:401|403)|\b(?:401|403)\b|write access to repository not granted)/i.test(
    message,
  );
}

/** Validates only an explicitly entered replacement token; stored credentials are never sent here. */
export function useGitTokenValidation(gitUrl: string, indexName?: string) {
  const [state, setState] = useState<GitTokenCheckState>('idle');
  const requestRef = useRef(0);
  const keyRef = useRef('');
  const pendingRef = useRef<{ key: string; request: number; promise: Promise<boolean> } | null>(
    null,
  );

  const invalidate = useCallback(() => {
    requestRef.current += 1;
    pendingRef.current = null;
    setState('idle');
  }, []);

  useEffect(() => invalidate(), [gitUrl, indexName, invalidate]);
  useEffect(
    () => () => {
      requestRef.current += 1;
    },
    [],
  );

  const validate = useCallback(
    async (rawToken: string): Promise<boolean> => {
      const token = rawToken.trim();
      if (!token || !gitUrl) {
        invalidate();
        return false;
      }
      const key = `${gitUrl}\u0000${indexName || ''}\u0000${token}`;
      if (state === 'valid' && keyRef.current === key) return true;
      if (pendingRef.current?.key === key) return pendingRef.current.promise;
      const request = requestRef.current + 1;
      requestRef.current = request;
      keyRef.current = key;
      setState('checking');
      const promise = (async () => {
        try {
          const result = await api.fetchBranches({
            git_url: gitUrl,
            git_token: token,
            index_name: indexName,
          });
          if (request !== requestRef.current || keyRef.current !== key) return false;
          const valid = !result.error && !result.needs_token;
          setState(valid ? 'valid' : result.needs_token ? 'invalid' : 'failed');
          return valid;
        } catch {
          if (request !== requestRef.current || keyRef.current !== key) return false;
          setState('failed');
          return false;
        } finally {
          if (pendingRef.current?.request === request) pendingRef.current = null;
        }
      })();
      pendingRef.current = { key, request, promise };
      return promise;
    },
    [gitUrl, indexName, invalidate, state],
  );

  return { state, validate, invalidate };
}
