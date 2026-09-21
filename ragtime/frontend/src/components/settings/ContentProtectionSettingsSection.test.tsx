import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ContentProtectionSettingsSection } from './ContentProtectionSettingsSection';

const { refreshModels } = vi.hoisted(() => ({ refreshModels: vi.fn() }));

vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => ({ models: [], loading: false, error: null, refresh: refreshModels }),
}));

const config = {
  revision: 3,
  enabled: true,
  classifier_model: null,
  coverage_mode: 'all_supported_traffic' as const,
  profiles: [{ id: 'standard', name: 'Standard', level: 0, scope: 'Operational content' }],
  group_profiles: [],
  requirements: [],
  user_overrides: [],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function renderSection() {
  return render(<ContentProtectionSettingsSection open onToggle={() => {}} />);
}

afterEach(() => {
  cleanup();
  refreshModels.mockClear();
  vi.unstubAllGlobals();
});

describe('ContentProtectionSettingsSection', () => {
  it('refreshes available models when opened, but not while closed', async () => {
    const { rerender } = render(
      <ContentProtectionSettingsSection open={false} onToggle={() => {}} />,
    );
    expect(refreshModels).not.toHaveBeenCalled();

    rerender(<ContentProtectionSettingsSection open onToggle={() => {}} />);
    await waitFor(() => expect(refreshModels).toHaveBeenCalledTimes(1));
  });

  it('keeps surface requirements editable in all supported traffic mode', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.endsWith('/config')) return Promise.resolve(json(config));
        if (url.endsWith('/catalog'))
          return Promise.resolve(
            json({
              users: [],
              groups: [],
              tools: [],
              mcp_routes: [],
              surfaces: [{ id: 'chat', name: 'Chat' }],
            }),
          );
        return Promise.resolve(json({ items: [] }));
      }),
    );
    renderSection();
    const requirement = await screen.findByLabelText('Coverage for Chat');
    expect((requirement as HTMLSelectElement).disabled).toBe(false);
    fireEvent.change(requirement, { target: { value: 'require' } });
    expect((requirement as HTMLSelectElement).value).toBe('require');
    expect(
      screen.getByText(
        'Coverage is currently All supported traffic; scope requirements apply when coverage is Selected scopes.',
      ),
    ).toBeTruthy();
  });

  it('sends the revisioned draft and explains a conflict without overwriting it', async () => {
    const fetch = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith('/config') && options?.method === 'PUT')
        return Promise.resolve(json({ detail: 'conflict' }, 409));
      if (url.endsWith('/config')) return Promise.resolve(json(config));
      if (url.endsWith('/catalog'))
        return Promise.resolve(
          json({ users: [], groups: [], tools: [], mcp_routes: [], surfaces: [] }),
        );
      return Promise.resolve(json({ items: [] }));
    });
    vi.stubGlobal('fetch', fetch);
    const user = userEvent.setup();
    renderSection();
    await screen.findByText('Operational content');
    await user.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toMatch(/changed on the server/i),
    );
    const saveCall = fetch.mock.calls.find(([, options]) => options?.method === 'PUT');
    expect(JSON.parse(saveCall?.[1]?.body as string)).toEqual({ expected_revision: 3, config });
  });

  it('previews the selected user, surface, route, and optional tool with server provenance', async () => {
    const fetch = vi.fn((url: string, _options?: RequestInit) => {
      if (url.endsWith('/config')) return Promise.resolve(json(config));
      if (url.endsWith('/catalog'))
        return Promise.resolve(
          json({
            users: [{ id: 'u1', name: 'Ada' }],
            groups: [],
            tools: [{ id: 'tool-1', name: 'Finance lookup' }],
            mcp_routes: [{ id: 'route-1', name: 'Finance MCP' }],
            surfaces: [{ id: 'workspace_chat', name: 'Workspace chat' }],
          }),
        );
      if (url.endsWith('/preview'))
        return Promise.resolve(
          json({
            required: true,
            provenance: 'user_override',
            profiles: [
              [config.profiles[0]],
              [{ id: 'finance', name: 'Finance', level: 1, scope: 'Financial records' }],
            ],
          }),
        );
      return Promise.resolve(json({ items: [] }));
    });
    vi.stubGlobal('fetch', fetch);
    const user = userEvent.setup();
    renderSection();
    await screen.findByText('Operational content');
    await user.selectOptions(screen.getByLabelText('User'), 'user:u1');
    await user.selectOptions(screen.getByLabelText('Surface'), 'workspace_chat');
    await user.selectOptions(screen.getByLabelText('MCP route'), 'route-1');
    await user.selectOptions(screen.getByLabelText('Tool'), 'tool-1');
    await user.click(screen.getByRole('button', { name: 'Preview effective policy' }));
    await waitFor(() => {
      const previewCall = fetch.mock.calls.find(([url]) => url.endsWith('/preview'));
      expect(JSON.parse(previewCall?.[1]?.body as string)).toEqual({
        user_id: 'u1',
        surface: 'workspace_chat',
        mcp_route: 'route-1',
        tool_id: 'tool-1',
        public: false,
      });
    });
    expect(
      await screen.findByText(
        /User: Ada.*Surface: workspace_chat.*MCP route: route-1.*Tool: tool-1.*Provenance: user_override.*Profile sets: Standard \/ Finance/,
      ),
    ).toBeTruthy();
  });

  it('shows a preview request failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.endsWith('/config')) return Promise.resolve(json(config));
        if (url.endsWith('/catalog'))
          return Promise.resolve(
            json({ users: [], groups: [], tools: [], mcp_routes: [], surfaces: [] }),
          );
        if (url.endsWith('/preview'))
          return Promise.resolve(json({ detail: 'Preview unavailable' }, 503));
        return Promise.resolve(json({ items: [] }));
      }),
    );
    const user = userEvent.setup();
    renderSection();
    await screen.findByText('Operational content');
    await user.click(screen.getByRole('button', { name: 'Preview effective policy' }));
    expect((await screen.findByRole('alert')).textContent).toContain('Preview unavailable');
  });

  it('does not create a broken mapping when deleting an assigned profile', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.endsWith('/config'))
          return Promise.resolve(
            json({ ...config, group_profiles: [{ group_id: 'group-1', profile_id: 'standard' }] }),
          );
        if (url.endsWith('/catalog'))
          return Promise.resolve(
            json({
              users: [],
              groups: [{ id: 'group-1', name: 'Finance' }],
              tools: [],
              mcp_routes: [],
              surfaces: [],
            }),
          );
        return Promise.resolve(json({ items: [] }));
      }),
    );
    const user = userEvent.setup();
    renderSection();
    await screen.findByRole('button', { name: 'Delete (1 groups)' });
    await user.click(screen.getByRole('button', { name: 'Delete (1 groups)' }));
    expect(screen.getByRole('alert').textContent).toMatch(/reassign or clear 1 affected group/i);
    expect(screen.getByLabelText('Standard scope')).toBeTruthy();
  });
});
