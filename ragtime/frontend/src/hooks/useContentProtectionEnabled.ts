import { useEffect, useRef, useState } from 'react';
import { contentProtectionApi } from '../api/contentProtection';

export type ContentProtectionEnabled = 'loading' | 'on' | 'off' | 'unknown';

export function useContentProtectionEnabled(active: boolean): ContentProtectionEnabled {
  const [status, setStatus] = useState<ContentProtectionEnabled>('loading');
  const request = useRef<Promise<ContentProtectionEnabled> | null>(null);

  useEffect(() => {
    if (!active) return;
    let mounted = true;
    request.current ??= contentProtectionApi
      .getConfig()
      .then((config): ContentProtectionEnabled => (config.enabled ? 'on' : 'off'))
      .catch((): ContentProtectionEnabled => 'unknown');
    void request.current.then((result) => {
      if (mounted) setStatus(result);
    });
    return () => {
      mounted = false;
    };
  }, [active]);

  return status;
}
