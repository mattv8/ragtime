import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => ({ listUserSpaceObjectStorageObjects: vi.fn() }));
vi.mock('@/api/client', () => ({ api: apiMock }));
vi.mock('../MountSourceWizard', () => ({ WorkspaceObjectStorageWizard: () => null }));
vi.mock('../DeleteConfirmButton', () => ({ DeleteConfirmButton: () => null }));
import { WorkspaceObjectStorageExplorer } from './WorkspaceObjectStorageExplorer';

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
describe('WorkspaceObjectStorageExplorer pagination', () => {
  it('sends continuation_token through the client when loading another page', async () => {
    apiMock.listUserSpaceObjectStorageObjects
      .mockResolvedValueOnce({
        workspace_id: 'w',
        bucket_name: 'files',
        prefix: '',
        entries: [
          { name: 'one', key: 'one', entry_type: 'object', size_bytes: 1, object_count: 0 },
        ],
        total_objects: null,
        total_bytes: null,
        next_continuation_token: 'cursor-2',
        is_truncated: true,
      })
      .mockResolvedValueOnce({
        workspace_id: 'w',
        bucket_name: 'files',
        prefix: '',
        entries: [
          { name: 'two', key: 'two', entry_type: 'object', size_bytes: 1, object_count: 0 },
        ],
        total_objects: null,
        total_bytes: null,
        next_continuation_token: null,
        is_truncated: false,
      });
    render(
      <WorkspaceObjectStorageExplorer
        workspaceId="w"
        loading={false}
        canManage={false}
        onConfigChange={() => {}}
        config={{
          workspace_id: 'w',
          region: 'us-east-1',
          endpoint_env_key: 'x',
          access_key_env_key: 'x',
          secret_key_env_key: 'x',
          public_object_search_paths: [],
          buckets: [
            {
              name: 'files',
              public_prefix: '',
              private_prefix: '',
              is_default: true,
              created_at: '',
              updated_at: '',
            },
          ],
        }}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: /files/i }));
    await screen.findByText('one');
    await userEvent.click(screen.getByRole('button', { name: /load more/i }));
    await waitFor(() =>
      expect(apiMock.listUserSpaceObjectStorageObjects).toHaveBeenLastCalledWith(
        'w',
        'files',
        '',
        'cursor-2',
      ),
    );
    expect(await screen.findByText('two')).toBeTruthy();
  });
});
