import type { SolidworksPdmConnectionConfig } from '../../types/api';
import { defaultScheduleStartMinute, defaultScheduleTimezone } from '../ScheduleStartTimeInput';

export const DEFAULT_PDM_CONNECTION_CONFIG: SolidworksPdmConnectionConfig = {
  host: '',
  port: 1433,
  user: '',
  password: '',
  database: '',
  file_extensions: ['.SLDPRT', '.SLDASM', '.SLDDRW'],
  exclude_deleted: true,
  variable_names: ['Description', 'Material', 'Revision', 'Status'],
  include_bom: true,
  include_folder_path: true,
  include_configurations: true,
  max_documents: null,
  reindex_interval_hours: 0,
  reindex_start_minute: null,
  reindex_timezone: null,
  last_indexed_at: null,
};

export function quickSetupPdmConfig(fields: {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
}): SolidworksPdmConnectionConfig {
  return {
    ...DEFAULT_PDM_CONNECTION_CONFIG,
    ...fields,
    reindex_interval_hours: 24,
    reindex_start_minute: defaultScheduleStartMinute(),
    reindex_timezone: defaultScheduleTimezone(),
  };
}
