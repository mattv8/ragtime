import { describe, expect, it } from 'vitest';

import type { MountProblem } from '@/types';
import {
  buildAdminMountWarnings,
  formatMountProblem,
  mountProblemSignature,
  mountRemediationHint,
  NON_ADMIN_MOUNT_WARNING,
} from './mountHealthWarnings';

const problem: MountProblem = {
  container: 'ragtime',
  mount_point: '/mnt/Accounting',
  fstype: 'cifs',
  source: '//192.168.10.3/Acc',
  state: 'failed',
  error: 'No such device',
  failing_since: '2026-10-02T11:53:00Z',
};

describe('mount health warnings', () => {
  it('formats a problem with relative failure time and error', () => {
    expect(formatMountProblem(problem, new Date('2026-10-02T12:00:00Z'))).toBe(
      '/mnt/Accounting (ragtime container) is unavailable since 7 min ago. Source: //192.168.10.3/Acc. Error: No such device.',
    );
  });

  it('uses the required relative time buckets and omits unknown failure time', () => {
    expect(
      formatMountProblem(
        { ...problem, container: 'runtime', state: 'unresponsive', error: null },
        new Date('2026-10-02T11:53:30Z'),
      ),
    ).toContain('(runtime container) is not responding since <1 min ago.');
    expect(
      formatMountProblem(
        { ...problem, failing_since: null, error: null },
        new Date('2026-10-02T12:00:00Z'),
      ),
    ).toBe('/mnt/Accounting (ragtime container) is unavailable. Source: //192.168.10.3/Acc.');
    expect(
      formatMountProblem(
        { ...problem, failing_since: 'not-a-date', error: null },
        new Date('2026-10-02T12:00:00Z'),
      ),
    ).toBe('/mnt/Accounting (ragtime container) is unavailable. Source: //192.168.10.3/Acc.');
    expect(
      formatMountProblem({ ...problem, failing_since: '2026-10-02T11:00:00Z' }, new Date('2026-10-02T12:00:00Z')),
    ).toContain('since 1 h ago');
    expect(
      formatMountProblem({ ...problem, failing_since: '2026-10-01T12:00:00Z' }, new Date('2026-10-02T12:00:00Z')),
    ).toContain('since 1 d ago');
  });

  it('builds admin warnings with remediation and a stable problem signature', () => {
    const runtimeProblem = {
      ...problem,
      container: 'runtime' as const,
      mount_point: '/mnt/Backup',
    };
    expect(buildAdminMountWarnings([problem], new Date('2026-10-02T12:00:00Z'))).toEqual([
      '/mnt/Accounting (ragtime container) is unavailable since 7 min ago. Source: //192.168.10.3/Acc. Error: No such device.',
      mountRemediationHint([problem]),
    ]);
    expect(mountProblemSignature([runtimeProblem, problem, problem])).toBe(
      'ragtime:/mnt/Accounting@2026-10-02T11:53:00Z|runtime:/mnt/Backup@2026-10-02T11:53:00Z',
    );
    expect(NON_ADMIN_MOUNT_WARNING).toBe(
      'Some shared files or workspace data may be temporarily unavailable. Contact an administrator if this persists.',
    );
  });

  it('uses automount and multi-mount remediation wording', () => {
    expect(
      formatMountProblem(
        { ...problem, fstype: 'autofs', source: 'host automount' },
        new Date('2026-10-02T11:53:30Z'),
      ),
    ).toBe(
      '/mnt/Accounting (ragtime container) is unavailable since <1 min ago. The host automount could not mount it. Error: No such device.',
    );
    expect(mountRemediationHint([problem, { ...problem, mount_point: '/mnt/Backup' }])).toContain(
      'for example: findmnt /mnt/Accounting; repeat for each listed path',
    );
  });
});
