import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelPreferencesModal } from './ModelPreferencesModal';

const apiMock = vi.hoisted(() => ({
  getModelPreferences: vi.fn(),
  updateModelPreference: vi.fn(),
  listUserSpaceWorkspaces: vi.fn(),
}));

const availableModelsMock = vi.hoisted(() => ({
  models: [
    {
      id: 'gpt-5',
      name: 'GPT-5',
      provider: 'openai',
      context_limit: 128000,
    },
    {
      id: 'claude-sonnet-4',
      name: 'Claude Sonnet 4',
      provider: 'anthropic',
      context_limit: 200000,
    },
  ],
  loading: false,
  error: null as string | null,
  readiness: null,
  meta: null,
  refresh: vi.fn(),
  awaitReady: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));
vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => availableModelsMock,
}));

function getGeneralSection(): HTMLElement {
  const section = document.getElementById('general-default-model-setting');
  if (!(section instanceof HTMLElement)) {
    throw new Error('Missing general default model section');
  }
  return section;
}

function getWorkspaceSection(): HTMLElement {
  const section = document.getElementById('workspace-default-model-setting');
  if (!(section instanceof HTMLElement)) {
    throw new Error('Missing workspace default model section');
  }
  return section;
}

describe('ModelPreferencesModal', () => {
  beforeEach(() => {
    apiMock.getModelPreferences.mockResolvedValue({
      user_default_chat_model: null,
      workspace_id: null,
      workspace_default_chat_model: null,
      global_default_chat_model: 'openai::gpt-5',
      effective_default_chat_model: 'openai::gpt-5',
    });
    apiMock.updateModelPreference.mockImplementation(
      async (model: string | null, workspaceId?: string) => ({
        user_default_chat_model: workspaceId ? 'anthropic::claude-sonnet-4' : model,
        workspace_id: workspaceId ?? null,
        workspace_default_chat_model: workspaceId ? model : null,
        global_default_chat_model: 'openai::gpt-5',
        effective_default_chat_model: model ?? 'openai::gpt-5',
      }),
    );
    apiMock.listUserSpaceWorkspaces
      .mockResolvedValueOnce({
        items: [
          {
            id: 'ws-1',
            name: 'Workspace One',
            sqlite_persistence_mode: 'exclude',
            owner_user_id: 'user-1',
            selected_tool_ids: [],
            selected_tool_group_ids: [],
            conversation_ids: [],
            members: [],
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
          },
          {
            id: 'ws-2',
            name: 'Workspace Two',
            sqlite_persistence_mode: 'exclude',
            owner_user_id: 'user-1',
            selected_tool_ids: [],
            selected_tool_group_ids: [],
            conversation_ids: [],
            members: [],
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
          },
        ],
        total: 3,
        offset: 0,
        limit: 50,
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'ws-3',
            name: 'Workspace Three',
            sqlite_persistence_mode: 'exclude',
            owner_user_id: 'user-1',
            selected_tool_ids: [],
            selected_tool_group_ids: [],
            conversation_ids: [],
            members: [],
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
          },
        ],
        total: 3,
        offset: 2,
        limit: 50,
      });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('loads the general preference and paginates through self-service workspaces', async () => {
    render(<ModelPreferencesModal isOpen onClose={vi.fn()} />);

    await screen.findByText('General default chat model');

    expect(apiMock.getModelPreferences).toHaveBeenCalledWith();
    await waitFor(() => {
      expect(apiMock.listUserSpaceWorkspaces).toHaveBeenNthCalledWith(1, 0, 50, false);
      expect(apiMock.listUserSpaceWorkspaces).toHaveBeenNthCalledWith(2, 2, 50, false);
    });
    expect(apiMock.listUserSpaceWorkspaces).toHaveBeenCalledTimes(2);
    expect(screen.getByText('Workspace One')).not.toBeNull();
    expect(screen.getByText('Workspace Three')).not.toBeNull();
    expect(screen.getByText('Inherited default: openai::gpt-5')).not.toBeNull();
  });

  it('portals the open modal overlay under document.body instead of the render container', async () => {
    const { container } = render(
      <div data-testid="render-shell">
        <ModelPreferencesModal isOpen onClose={vi.fn()} />
      </div>,
    );

    await screen.findByText('General default chat model');

    const overlay = document.querySelector('.modal-overlay');
    if (!(overlay instanceof HTMLElement)) {
      throw new Error('Expected modal overlay');
    }

    expect(container.contains(overlay)).toBe(false);
    expect(document.body.contains(overlay)).toBe(true);
    expect(overlay.parentElement).toBe(document.body);
  });

  it('uses provider-scoped keys for persisted selection and saves scoped model identifiers', async () => {
    const user = userEvent.setup();
    apiMock.getModelPreferences.mockResolvedValueOnce({
      user_default_chat_model: 'openai::gpt-5',
      workspace_id: null,
      workspace_default_chat_model: null,
      global_default_chat_model: 'openai::gpt-5',
      effective_default_chat_model: 'openai::gpt-5',
    });
    apiMock.updateModelPreference.mockResolvedValueOnce({
      user_default_chat_model: 'anthropic::claude-sonnet-4',
      workspace_id: null,
      workspace_default_chat_model: null,
      global_default_chat_model: 'openai::gpt-5',
      effective_default_chat_model: 'anthropic::claude-sonnet-4',
    });

    render(<ModelPreferencesModal isOpen onClose={vi.fn()} />);

    const generalSection = await waitFor(() => getGeneralSection());
    expect(
      within(generalSection).getByRole('button', { name: /openai gpt-5/i }),
    ).not.toBeNull();

    await user.click(within(generalSection).getByRole('button', { name: /openai gpt-5/i }));
    await user.type(screen.getByRole('textbox', { name: 'Filter models' }), 'claude');
    await user.click(screen.getByRole('button', { name: /anthropic claude sonnet 4/i }));
    await user.click(within(generalSection).getByRole('button', { name: 'Save general default' }));

    await waitFor(() => {
      expect(apiMock.updateModelPreference).toHaveBeenCalledWith('anthropic::claude-sonnet-4');
    });
  });

  it('saves and resets the general preference', async () => {
    const user = userEvent.setup();
    render(<ModelPreferencesModal isOpen onClose={vi.fn()} />);

    const generalSection = await waitFor(() => getGeneralSection());
    await user.click(within(generalSection).getByRole('button', { name: /openai gpt-5/i }));
    await user.type(screen.getByRole('textbox', { name: 'Filter models' }), 'claude');
    await user.click(screen.getByRole('button', { name: /anthropic claude sonnet 4/i }));
    await user.click(within(generalSection).getByRole('button', { name: 'Save general default' }));

    await waitFor(() => {
      expect(apiMock.updateModelPreference).toHaveBeenCalledWith('anthropic::claude-sonnet-4');
    });
    expect(screen.getByText('Personal override saved.')).not.toBeNull();

    await user.click(screen.getByRole('button', { name: 'Reset general default' }));
    await waitFor(() => {
      expect(apiMock.updateModelPreference).toHaveBeenLastCalledWith(null);
    });
    expect(screen.getByText('Inherited default: openai::gpt-5')).not.toBeNull();
  });

  it('loads, saves, and resets a workspace preference while showing inherited values', async () => {
    const user = userEvent.setup();
    apiMock.getModelPreferences
      .mockResolvedValueOnce({
        user_default_chat_model: 'anthropic::claude-sonnet-4',
        workspace_id: null,
        workspace_default_chat_model: null,
        global_default_chat_model: 'openai::gpt-5',
        effective_default_chat_model: 'anthropic::claude-sonnet-4',
      })
      .mockResolvedValueOnce({
        user_default_chat_model: 'anthropic::claude-sonnet-4',
        workspace_id: 'ws-2',
        workspace_default_chat_model: null,
        global_default_chat_model: 'openai::gpt-5',
        effective_default_chat_model: 'anthropic::claude-sonnet-4',
      });
    apiMock.updateModelPreference.mockResolvedValueOnce({
      user_default_chat_model: 'anthropic::claude-sonnet-4',
      workspace_id: 'ws-2',
      workspace_default_chat_model: 'openai::gpt-5',
      global_default_chat_model: 'openai::gpt-5',
      effective_default_chat_model: 'openai::gpt-5',
    });
    apiMock.updateModelPreference.mockResolvedValueOnce({
      user_default_chat_model: 'anthropic::claude-sonnet-4',
      workspace_id: 'ws-2',
      workspace_default_chat_model: null,
      global_default_chat_model: 'openai::gpt-5',
      effective_default_chat_model: 'anthropic::claude-sonnet-4',
    });

    render(<ModelPreferencesModal isOpen onClose={vi.fn()} />);

    const workspaceSection = await waitFor(() => getWorkspaceSection());
    const picker = within(workspaceSection).getByLabelText('Workspace');
    await user.selectOptions(picker, 'ws-2');

    await waitFor(() => {
      expect(apiMock.getModelPreferences).toHaveBeenLastCalledWith('ws-2');
    });
    expect(screen.getByText('Inherited default: anthropic::claude-sonnet-4')).not.toBeNull();

    await user.click(within(workspaceSection).getByRole('button', { name: /anthropic claude sonnet 4/i }));
    await user.type(screen.getByRole('textbox', { name: 'Filter models' }), 'gpt');
    await user.click(screen.getByRole('button', { name: /openai gpt-5/i }));
    await user.click(within(workspaceSection).getByRole('button', { name: 'Save workspace default' }));

    await waitFor(() => {
      expect(apiMock.updateModelPreference).toHaveBeenCalledWith('openai::gpt-5', 'ws-2');
    });

    await user.click(screen.getByRole('button', { name: 'Reset workspace default' }));
    await waitFor(() => {
      expect(apiMock.updateModelPreference).toHaveBeenLastCalledWith(null, 'ws-2');
    });
    expect(screen.getByText('Inherited default: anthropic::claude-sonnet-4')).not.toBeNull();
  });

  it('shows no-workspace and API-error states', async () => {
    apiMock.listUserSpaceWorkspaces.mockReset();
    apiMock.listUserSpaceWorkspaces.mockResolvedValue({ items: [], total: 0, offset: 0, limit: 50 });
    apiMock.getModelPreferences.mockRejectedValueOnce(new Error('General preferences failed'));
    render(<ModelPreferencesModal isOpen onClose={vi.fn()} />);

    await screen.findByText('General preferences failed');
    await screen.findByText('No workspaces available for workspace-specific defaults.');
  });

  it('shows workspace load errors and disables selectors while models are loading', async () => {
    const user = userEvent.setup();
    availableModelsMock.loading = true;
    apiMock.getModelPreferences
      .mockResolvedValueOnce({
        user_default_chat_model: null,
        workspace_id: null,
        workspace_default_chat_model: null,
        global_default_chat_model: 'openai::gpt-5',
        effective_default_chat_model: 'openai::gpt-5',
      })
      .mockRejectedValueOnce(new Error('Workspace preferences failed'));

    render(<ModelPreferencesModal isOpen onClose={vi.fn()} />);

    const generalSection = await waitFor(() => getGeneralSection());
    expect(within(generalSection).getByRole('button', { name: /openai gpt-5/i })).toHaveProperty(
      'disabled',
      true,
    );
    const picker = await screen.findByLabelText('Workspace');
    await user.selectOptions(picker, 'ws-1');
    await screen.findByText('Workspace preferences failed');
    availableModelsMock.loading = false;
  });

  it('keeps reset available when the catalog is unavailable but an override exists', async () => {
    availableModelsMock.models = [];
    availableModelsMock.error = 'Catalog unavailable';
    apiMock.getModelPreferences.mockResolvedValueOnce({
      user_default_chat_model: 'openai::gpt-5',
      workspace_id: null,
      workspace_default_chat_model: null,
      global_default_chat_model: 'openai::gpt-5',
      effective_default_chat_model: 'openai::gpt-5',
    });

    render(<ModelPreferencesModal isOpen onClose={vi.fn()} />);

    const generalSection = await waitFor(() => getGeneralSection());
    expect(
      within(generalSection).getByRole('button', { name: 'Reset general default' }),
    ).toHaveProperty('disabled', false);
    expect(
      within(generalSection).getByRole('button', { name: 'Save general default' }),
    ).toHaveProperty('disabled', true);

    availableModelsMock.models = [
      {
        id: 'gpt-5',
        name: 'GPT-5',
        provider: 'openai',
        context_limit: 128000,
      },
      {
        id: 'claude-sonnet-4',
        name: 'Claude Sonnet 4',
        provider: 'anthropic',
        context_limit: 200000,
      },
    ];
    availableModelsMock.error = null;
  });
});
