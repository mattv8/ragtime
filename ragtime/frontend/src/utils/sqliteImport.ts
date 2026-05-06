export const SQLITE_IMPORT_MIN_BYTES = 100 * 1024 * 1024;
export const SQLITE_IMPORT_DEFAULT_MAX_BYTES = SQLITE_IMPORT_MIN_BYTES;
export const SQLITE_IMPORT_MAX_BYTES = 100 * 1024 * 1024 * 1024;

const SQLITE_IMPORT_SLIDER_SCALE = Math.log(SQLITE_IMPORT_MAX_BYTES / SQLITE_IMPORT_MIN_BYTES);

export function sqliteImportBytesToSlider(bytes: number): number {
  if (bytes >= SQLITE_IMPORT_MAX_BYTES) return 100;
  if (bytes <= SQLITE_IMPORT_MIN_BYTES) return 0;
  return Math.max(
    0,
    Math.min(100, (Math.log(bytes / SQLITE_IMPORT_MIN_BYTES) / SQLITE_IMPORT_SLIDER_SCALE) * 100),
  );
}

export function sliderToSqliteImportBytes(slider: number): number {
  if (slider >= 100) return SQLITE_IMPORT_MAX_BYTES;
  if (slider <= 0) return SQLITE_IMPORT_MIN_BYTES;
  return Math.round(SQLITE_IMPORT_MIN_BYTES * Math.exp((slider / 100) * SQLITE_IMPORT_SLIDER_SCALE));
}
