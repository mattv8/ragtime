const NON_PREFETCHABLE_USER_SPACE_FILE_EXTENSIONS = [
  '.sqlite',
  '.sqlite3',
  '.db',
  '.db-wal',
  '.db-shm',
  '.sqlite-wal',
  '.sqlite-shm',
  // images
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.bmp',
  '.tiff',
  '.ico',
  // fonts
  '.woff',
  '.woff2',
  '.ttf',
  '.otf',
  '.eot',
  // archives
  '.zip',
  '.tar',
  '.gz',
  '.bz2',
  '.7z',
  '.rar',
  // document and office formats that are opened through specialized flows, not the text editor
  '.pdf',
  '.doc',
  '.docx',
  '.docm',
  '.ppt',
  '.pptx',
  '.pptm',
  '.pps',
  '.ppsx',
  '.ppsm',
  '.pot',
  '.xls',
  '.xlsx',
  '.xlsm',
  '.xlsb',
  '.odt',
  '.ods',
  '.odp',
  '.rtf',
  '.epub',
  '.msg',
  // binary/tabular data formats
  '.parquet',
  '.feather',
  '.avro',
];

export const WARM_CACHE_MAX_FILE_BYTES = 512 * 1024;
export const WARM_CACHE_PREFETCH_CONCURRENCY = 4;

export interface WarmCacheFileInfo {
  path: string;
  size_bytes?: number | null;
  updated_at?: string | null;
}

export interface WarmCacheEntryInfo {
  updatedAt: string;
}

// Deny-list semantics: unknown extensions are treated as prefetchable so text
// files without a known extension still warm the editor cache.
export function isPrefetchableTextUserSpaceFilePath(path: string): boolean {
  const lower = path.toLowerCase();
  return !NON_PREFETCHABLE_USER_SPACE_FILE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export function getWarmCacheCandidateFiles<
  TFile extends WarmCacheFileInfo,
  TCache extends WarmCacheEntryInfo,
>(
  files: TFile[],
  cache: Record<string, TCache | undefined>,
  options?: { excludePaths?: string[] },
): TFile[] {
  const excludedPaths = new Set(options?.excludePaths ?? []);
  return files.filter((file) => {
    if (excludedPaths.has(file.path)) {
      return false;
    }
    if (!isPrefetchableTextUserSpaceFilePath(file.path)) {
      return false;
    }
    if ((file.size_bytes ?? 0) > WARM_CACHE_MAX_FILE_BYTES) {
      return false;
    }
    const cached = cache[file.path];
    return !cached || cached.updatedAt !== (file.updated_at ?? '');
  });
}

export async function runWithConcurrencyLimit<TItem, TResult>(
  items: TItem[],
  limit: number,
  worker: (item: TItem, index: number) => Promise<TResult>,
): Promise<TResult[]> {
  const boundedLimit = Math.max(1, Math.floor(limit));
  const results = new Array<TResult>(items.length);
  let nextIndex = 0;
  let aborted = false;

  async function runNext(): Promise<void> {
    while (!aborted && nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      try {
        results[index] = await worker(items[index], index);
      } catch (error) {
        aborted = true;
        throw error;
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(boundedLimit, items.length) }, () => runNext()));
  return results;
}
