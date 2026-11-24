import type { ToolConfig } from '@/types';

export function formatToolCount(n: number): string {
  return new Intl.NumberFormat().format(n);
}

export function formatToolFactDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function hasSchemaIndexingEnabled(tool: ToolConfig): boolean {
  return (
    (tool.tool_type === 'postgres' || tool.tool_type === 'mssql' || tool.tool_type === 'mysql') &&
    (tool.connection_config as { schema_index_enabled?: boolean })?.schema_index_enabled === true
  );
}

export function getToolWorkingDirectory(tool: ToolConfig): string | null {
  if (
    tool.tool_type !== 'ssh_shell' &&
    tool.tool_type !== 'windows_shell' &&
    tool.tool_type !== 'odoo_shell'
  ) {
    return null;
  }

  return (tool.connection_config as { working_directory?: string })?.working_directory || null;
}

export function isPairedWindowsTool(tool: ToolConfig): boolean {
  return (
    (tool.tool_type === 'windows_shell' || tool.tool_type === 'solidworks') &&
    typeof (tool.connection_config as { pairing_id?: unknown }).pairing_id === 'string' &&
    (tool.connection_config as { pairing_id: string }).pairing_id.length > 0
  );
}

export function getToolConnectionSummary(tool: ToolConfig): string {
  const config = tool.connection_config;
  switch (tool.tool_type) {
    case 'postgres':
    case 'mysql':
      if ('host' in config && config.host) {
        const port = 'port' in config ? config.port : tool.tool_type === 'mysql' ? 3306 : 5432;
        const database = 'database' in config ? config.database : '';
        return `${config.host}:${port}/${database}`;
      }
      return `Container: ${'container' in config ? config.container : 'N/A'}`;
    case 'mssql':
    case 'solidworks_pdm':
      if ('host' in config && config.host) {
        const port = 'port' in config ? config.port : 1433;
        const database = 'database' in config ? config.database : '';
        return `${config.host}:${port}/${database}`;
      }
      return 'MSSQL connection';
    case 'influxdb':
      if ('host' in config && config.host) {
        const port = 'port' in config ? config.port : 8086;
        const scheme = 'use_https' in config && config.use_https ? 'https' : 'http';
        const bucket = 'bucket' in config && config.bucket ? config.bucket : '(no bucket)';
        return `${scheme}://${config.host}:${port}/${bucket}`;
      }
      return 'InfluxDB connection';
    case 'odoo_shell':
      if ('mode' in config && config.mode === 'ssh') {
        if ('ssh_host' in config && 'ssh_user' in config) {
          const port = 'ssh_port' in config ? config.ssh_port : 22;
          return `${config.ssh_user}@${config.ssh_host}:${port}`;
        }
        return 'SSH connection';
      }
      return `Container: ${'container' in config ? config.container : 'N/A'}`;
    case 'ssh_shell':
    case 'windows_shell':
    case 'solidworks':
      if ('host' in config && 'user' in config) {
        const port = 'port' in config ? config.port : 22;
        return `${config.user}@${config.host}:${port}`;
      }
      return 'SSH connection';
    case 'filesystem_indexer':
      if ('paths' in config && Array.isArray(config.paths)) {
        return `${config.paths.length} path(s)`;
      }
      return 'Filesystem indexer';
    case 'http_api': {
      const baseUrl = 'base_url' in config && config.base_url ? config.base_url : 'HTTP API';
      const authMode = 'auth_mode' in config && config.auth_mode ? config.auth_mode : 'none';
      return `${baseUrl} (${authMode})`;
    }
    default:
      return 'Unknown';
  }
}
