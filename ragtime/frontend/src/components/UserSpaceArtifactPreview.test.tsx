import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  USERSPACE_EXEC_BRIDGE,
  USERSPACE_EXEC_MESSAGE_TYPES,
} from '@/utils/userspacePreview/constants';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    getUserSpacePreviewSettings: vi.fn(),
  },
}));

vi.mock('@/api/client', () => ({ api: apiMock }));

import { UserSpaceArtifactPreview } from './UserSpaceArtifactPreview';

describe('UserSpaceArtifactPreview', () => {
  beforeEach(() => {
    apiMock.getUserSpacePreviewSettings.mockResolvedValue({
      userspace_preview_sandbox_flags: ['allow-scripts'],
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('surfaces execution errors through callbacks without rendering legacy preview banners', async () => {
    const onLiveDataWarningChange = vi.fn();
    const onLiveDataTimeout = vi.fn();
    const onPreviewOverlayMessage = vi.fn();

    render(
      <UserSpaceArtifactPreview
        entryPath="dashboard/main.ts"
        workspaceFiles={{ 'dashboard/main.ts': 'export const ready = true;' }}
        runtimePreviewUrl="http://preview.test/session"
        runtimePreviewOrigin="http://preview.test"
        runtimeAvailable
        workspaceId="ws-1"
        onLiveDataWarningChange={onLiveDataWarningChange}
        onLiveDataTimeout={onLiveDataTimeout}
        onPreviewOverlayMessage={onPreviewOverlayMessage}
      />,
    );

    const iframe = (await screen.findByTitle('Runtime preview')) as HTMLIFrameElement;
    const frameWindow = iframe.contentWindow;
    expect(frameWindow).not.toBeNull();

    window.dispatchEvent(
      new MessageEvent('message', {
        origin: 'http://preview.test',
        source: frameWindow ?? undefined,
        data: {
          bridge: USERSPACE_EXEC_BRIDGE,
          type: USERSPACE_EXEC_MESSAGE_TYPES.ERROR,
          component_id: 'sales-chart',
          error: 'statement timeout',
          error_kind: 'timeout',
          timeout_seconds: 12,
        },
      }),
    );

    await waitFor(() => {
      expect(onLiveDataWarningChange).toHaveBeenCalledWith('statement timeout');
    });

    expect(onLiveDataTimeout).toHaveBeenCalledWith('statement timeout', 12);
    expect(onPreviewOverlayMessage).not.toHaveBeenCalled();
    expect(document.querySelector('.userspace-preview-exec-notice')).toBeNull();
    expect(document.querySelector('.userspace-preview-exec-error')).toBeNull();
  });
});
