import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ToolWizard } from './ToolWizard';
import type { ToolConfig } from '@/types';

const apiMock = vi.hoisted(() => ({
  discoverDocker: vi.fn(),
  connectToNetwork: vi.fn(),
  discoverPostgresDatabases: vi.fn(),
  discoverMssqlDatabases: vi.fn(),
  discoverMysqlDatabases: vi.fn(),
  discoverInfluxdbBuckets: vi.fn(),
  generateSSHKeypair: vi.fn(),
  updateToolConfig: vi.fn(),
  testToolConnection: vi.fn(),
  createToolConfig: vi.fn(),
  startFilesystemAnalysis: vi.fn(),
  getFilesystemAnalysisJob: vi.fn(),
  triggerSchemaIndex: vi.fn(),
  triggerPdmIndex: vi.fn(),
  discoverMounts: vi.fn(),
  browseFilesystem: vi.fn(),
  browseSSHFilesystem: vi.fn(),
  discoverNfsExports: vi.fn(),
  browseNfsExport: vi.fn(),
  discoverSmbShares: vi.fn(),
  browseSmbShare: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

const postgresContainerTool: ToolConfig = {
  id: 'tool-postgres-container',
  name: 'Container Postgres',
  tool_type: 'postgres',
  enabled: true,
  description: 'PostgreSQL in Docker',
  connection_config: {
    container: 'postgres-1',
    docker_network: '',
    docker_ssh_enabled: false,
    database: '',
    host: '',
    port: 5432,
    user: '',
    password: '',
  },
  max_results: 100,
  timeout_max_seconds: 300,
  allow_write: false,
  sort_order: 100,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ToolWizard remote Docker SSH layout', () => {
  it('groups SSH host, port, and user in one compact row with a flat auth panel', () => {
    const { container } = render(
      <ToolWizard existingTool={postgresContainerTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );

    fireEvent.click(screen.getByLabelText('Remote Docker host via SSH'));

    const remoteSshRow = container.querySelector('.remote-docker-ssh-row');
    expect(remoteSshRow).not.toBeNull();
    expect(remoteSshRow?.querySelectorAll('input')).toHaveLength(3);
    const rowLabels = Array.from(remoteSshRow?.querySelectorAll('label') ?? []).map((label) =>
      label.textContent?.trim(),
    );
    expect(rowLabels).toEqual(['SSH Host', 'SSH Port', 'SSH User']);

    expect(container.querySelector('.ssh-auth-panel.compact')).not.toBeNull();
    expect(container.querySelector('.ssh-key-panel.flat')).not.toBeNull();
  });
});
