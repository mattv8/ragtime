import { describe, expect, it } from 'vitest';

import { getWarmCacheCandidateFiles, runWithConcurrencyLimit } from './userspacePrefetch';

describe('getWarmCacheCandidateFiles', () => {
  it.each([
    {
      label: 'unsupported binary and office files',
      files: [
        { path: 'dashboard/main.ts', size_bytes: 100, updated_at: '1' },
        { path: 'reconciliations/11101.xlsx', size_bytes: 100, updated_at: '1' },
        { path: 'reconciliations/21407.xlsm', size_bytes: 100, updated_at: '1' },
        { path: 'docs/brief.docx', size_bytes: 100, updated_at: '1' },
        { path: 'slides/report.pptx', size_bytes: 100, updated_at: '1' },
      ],
      cache: {},
      options: undefined,
      expectedPaths: ['dashboard/main.ts'],
    },
    {
      label: 'case-insensitive and compound skipped extensions',
      files: [
        { path: 'reports/REPORT.XLSX', size_bytes: 100, updated_at: '1' },
        { path: 'archives/source.tar.gz', size_bytes: 100, updated_at: '1' },
        { path: 'notes/README', size_bytes: 100, updated_at: '1' },
      ],
      cache: {},
      options: undefined,
      expectedPaths: ['notes/README'],
    },
    {
      label: 'unchanged, excluded, and oversized files',
      files: [
        { path: 'dashboard/current.ts', size_bytes: 100, updated_at: '1' },
        { path: 'dashboard/excluded.ts', size_bytes: 100, updated_at: '1' },
        { path: 'server/large.mjs', size_bytes: 513 * 1024, updated_at: '1' },
      ],
      cache: {
        'dashboard/current.ts': { content: 'cached', updatedAt: '1', artifactType: null },
      },
      options: { excludePaths: ['dashboard/excluded.ts'] },
      expectedPaths: [],
    },
    {
      label: 'changed cached files and null metadata defaults',
      files: [
        { path: 'dashboard/changed.ts', size_bytes: 100, updated_at: '2' },
        { path: 'dashboard/null-metadata.ts', size_bytes: null, updated_at: null },
        { path: 'dashboard/null-unchanged.ts', size_bytes: null, updated_at: null },
      ],
      cache: {
        'dashboard/changed.ts': { content: 'cached', updatedAt: '1', artifactType: null },
        'dashboard/null-unchanged.ts': { content: 'cached', updatedAt: '', artifactType: null },
      },
      options: undefined,
      expectedPaths: ['dashboard/changed.ts', 'dashboard/null-metadata.ts'],
    },
  ])('filters warm-cache candidates for $label', ({ files, cache, options, expectedPaths }) => {
    const candidates = getWarmCacheCandidateFiles(files, cache, options);

    expect(candidates.map((file) => file.path)).toEqual(expectedPaths);
  });
});

describe('runWithConcurrencyLimit', () => {
  it('returns an empty result for an empty item list', async () => {
    await expect(runWithConcurrencyLimit([], 2, async (item) => item)).resolves.toEqual([]);
  });

  it('clamps invalid limits to one worker', async () => {
    let active = 0;
    let maxActive = 0;

    const results = await runWithConcurrencyLimit([1, 2, 3], 0, async (item) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return item;
    });

    expect(maxActive).toBe(1);
    expect(results).toEqual([1, 2, 3]);
  });

  it('bounds concurrent workers while preserving result order', async () => {
    let active = 0;
    let maxActive = 0;

    const results = await runWithConcurrencyLimit([1, 2, 3, 4, 5], 2, async (item) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return item * 10;
    });

    expect(maxActive).toBeLessThanOrEqual(2);
    expect(results).toEqual([10, 20, 30, 40, 50]);
  });

  it('rejects when a worker rejects', async () => {
    await expect(
      runWithConcurrencyLimit([1, 2, 3], 2, async (item) => {
        if (item === 2) {
          throw new Error('prefetch failed');
        }
        return item;
      }),
    ).rejects.toThrow('prefetch failed');
  });

  it('stops scheduling new workers after a rejection', async () => {
    const started: number[] = [];

    await expect(
      runWithConcurrencyLimit([1, 2, 3], 2, async (item) => {
        started.push(item);
        if (item === 1) {
          await new Promise((resolve) => setTimeout(resolve, 5));
        }
        if (item === 2) {
          throw new Error('prefetch failed');
        }
        return item;
      }),
    ).rejects.toThrow('prefetch failed');

    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(started).toEqual([1, 2]);
  });
});
