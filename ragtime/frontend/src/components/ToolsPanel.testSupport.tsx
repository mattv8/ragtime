import { cleanup } from '@testing-library/react';
import type { HTMLAttributes, ReactNode } from 'react';
import { vi } from 'vitest';
import type { InstructionSkill, InstructionSkillOwner, ToolConfig, ToolGroup } from '@/types';

export const apiMock = {
  listToolConfigs: vi.fn(),
  listToolGroups: vi.fn(),
  listUserspaceMountSources: vi.fn(),
  getSettings: vi.fn(),
  getToolHeartbeats: vi.fn(),
  subscribeToolHealthEvents: vi.fn(),
  updateToolConfig: vi.fn(),
  toggleToolConfig: vi.fn(),
  deleteToolConfig: vi.fn(),
  createToolConfig: vi.fn(),
  reorderTools: vi.fn(),
  clearToolUndecryptableCredentials: vi.fn(),
  getPdmIndexStats: vi.fn(),
  getToolAccessPolicy: vi.fn(),
  updateToolAccessPolicy: vi.fn(),
  listUsers: vi.fn(),
  listUsersDirectory: vi.fn(),
  listAuthGroups: vi.fn(),
  getInstructionSkillSummary: vi.fn(),
  copyToolInstructionSkills: vi.fn(),
};

export const skillsDialogMock = {
  nextSkills: [] as InstructionSkill[],
};

export const toastMock = {
  success: vi.fn(),
  info: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
};

export const toolFilterState = {
  queries: [] as string[],
  tags: [] as string[],
  input: '',
  debouncedInput: '',
  hasActiveFilters: false,
  setInput: vi.fn(),
  setTags: vi.fn(),
  clear: vi.fn(),
};

export const resetToolFilterState = (): void => {
  toolFilterState.queries = [];
  toolFilterState.tags = [];
  toolFilterState.input = '';
  toolFilterState.debouncedInput = '';
  toolFilterState.hasActiveFilters = false;
};

export const contentProtectionMock = {
  getConfig: vi.fn(),
  updateContentProtectionConfigSlice: vi.fn(),
};
export const wizardState: { existingTool: ToolConfig | null | undefined } = {
  existingTool: undefined,
};

export const toolWizardMock = {
  ToolWizard: (props: { existingTool: ToolConfig | null }) => {
    wizardState.existingTool = props.existingTool;
    return null;
  },
};
export const mountSourceWizardMock = { MountSourceWizard: () => null };
export const instructionSkillsDialogMock = {
  InstructionSkillsDialog: ({
    open,
    owner,
    ownerName,
    onClose,
    onSkillsChanged,
  }: {
    open: boolean;
    owner: InstructionSkillOwner;
    ownerName: string;
    onClose: () => void;
    onSkillsChanged?: (skills: InstructionSkill[]) => void;
  }) =>
    open ? (
      <div
        role="dialog"
        aria-label={`Skills · ${ownerName}`}
        data-owner-kind={owner.kind}
        data-owner-id={owner.id}
      >
        <button type="button" onClick={onClose}>
          Close Skills Mock
        </button>
        <button type="button" onClick={() => onSkillsChanged?.(skillsDialogMock.nextSkills)}>
          Save Skills Mock
        </button>
      </div>
    ) : null,
};
export const toolAccessModalMock = {
  ToolAccessModal: ({
    open,
    toolName,
    onSave,
  }: {
    open: boolean;
    toolName: string;
    onSave: (policy: unknown) => void;
  }) =>
    open ? (
      <div>
        <h3>Tool Access {toolName}</h3>
        <button
          type="button"
          onClick={() =>
            onSave({
              tool_id: 'tool-grouped',
              default_chat_access: 'read',
              default_workspace_access: 'deny',
              users: [],
              groups: [],
            })
          }
        >
          Save Access
        </button>
      </div>
    ) : null,
};

type MockPopoverProps = {
  children: ReactNode;
  content?: ReactNode;
  position?: string;
  show?: boolean;
  trigger?: string;
  disabled?: boolean;
  openDelayMs?: number;
  followCursor?: boolean;
  requireHoverIdleMs?: number;
  focusTrigger?: boolean;
  ignoreSelector?: string;
} & HTMLAttributes<HTMLDivElement>;

export const popoverMock = {
  Popover: ({
    children,
    content: _content,
    position,
    show: _show,
    trigger: _trigger,
    disabled: _disabled,
    openDelayMs: _openDelayMs,
    followCursor,
    requireHoverIdleMs,
    focusTrigger,
    ignoreSelector,
    ...rest
  }: MockPopoverProps) => (
    <div
      {...rest}
      data-follow-cursor={followCursor === undefined ? undefined : String(followCursor)}
      data-require-hover-idle-ms={requireHoverIdleMs ?? undefined}
      data-focus-trigger={focusTrigger === undefined ? undefined : String(focusTrigger)}
      data-position={position}
      data-popover-content={typeof _content === 'string' ? _content : undefined}
      data-ignore-selector={ignoreSelector}
    >
      {children}
    </div>
  ),
};
export const animatedCreateButtonMock = {
  AnimatedCreateButton: ({ onClick, label }: { onClick: () => void; label: string }) => (
    <button type="button" onClick={onClick}>
      {label}
    </button>
  ),
};
export const deleteConfirmButtonMock = {
  DeleteConfirmButton: ({
    onDelete,
    className,
    title,
  }: {
    onDelete: () => void;
    className?: string;
    title?: string;
  }) => (
    <button
      type="button"
      className={className}
      title={title}
      data-delete-confirm-button
      onClick={onDelete}
    >
      Delete
    </button>
  ),
};
export const toastComponentsMock = {
  useToast: () => [[], toastMock],
  ToastContainer: () => null,
};
export const searchFilterBarMock = {
  SearchFilterBar: () => null,
  searchFilterTextMatchesQuery: () => true,
  useUrlSearchFilterState: () => toolFilterState,
};

export const groupedTool: ToolConfig = {
  id: 'tool-grouped',
  name: 'Grouped Tool',
  tool_type: 'ssh_shell',
  enabled: true,
  description: 'Grouped tool',
  connection_config: { host: 'example.com', user: 'deploy', port: 22 },
  max_results: 10,
  timeout_max_seconds: 30,
  allow_write: false,
  sort_order: 100,
  group_id: 'group-1',
  group_name: 'Alpha Group',
  undecryptable_fields: [],
  configured_secret_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

export const ungroupedTool: ToolConfig = {
  id: 'tool-ungrouped',
  name: 'Ungrouped Tool',
  tool_type: 'ssh_shell',
  enabled: true,
  description: 'Ungrouped tool',
  connection_config: { host: 'example.org', user: 'deploy', port: 22 },
  max_results: 10,
  timeout_max_seconds: 30,
  allow_write: false,
  sort_order: 200,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

export const pdmTool: ToolConfig = {
  id: 'tool-pdm',
  name: 'PDM Tool',
  tool_type: 'solidworks_pdm',
  enabled: true,
  description: 'SolidWorks PDM database',
  connection_config: {
    host: '192.168.10.12',
    port: 1433,
    user: 'pdm-user',
    password: 'secret',
    database: 'PDM',
  },
  max_results: 10,
  timeout_max_seconds: 30,
  allow_write: false,
  sort_order: 300,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

export const httpApiTool: ToolConfig = {
  id: 'tool-http-api',
  name: 'Orders API',
  tool_type: 'http_api',
  enabled: true,
  description: 'HTTP API connection',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'bearer',
  },
  max_results: 50,
  timeout_max_seconds: 300,
  allow_write: false,
  sort_order: 400,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: ['bearer_token'],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

export const pairedWindowsTool: ToolConfig = {
  ...ungroupedTool,
  id: 'tool-windows-paired',
  name: 'Paired Windows Tool',
  tool_type: 'windows_shell',
  connection_config: {
    host: 'windows.example.org',
    user: 'administrator',
    port: 22,
    pairing_id: 'a0a0a0a0-0000-4000-8000-000000000000',
  },
};

export const linkedSolidworksTool: ToolConfig = {
  ...ungroupedTool,
  id: 'tool-solidworks',
  name: 'CAD workstation',
  tool_type: 'solidworks',
  connection_config: {
    host: 'cad.example.org',
    user: 'cad',
    port: 22,
    pdm_tool_id: pdmTool.id,
    pdm_approved_folders: ['\\Designs', '\\Parts'],
  },
};

export const toolGroup: ToolGroup = {
  id: 'group-1',
  name: 'Alpha Group',
  description: '',
  sort_order: 100,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

export function skillFixture(id: string, enabled: boolean): InstructionSkill {
  return {
    id,
    owner_kind: 'tool',
    owner_id: 'tool-ungrouped',
    name: `skill-${id}`,
    description: 'Describes the skill',
    content: 'Skill body',
    mode: 'on_demand',
    enabled,
    sort_order: 0,
    uri: `skill://tools/tool-ungrouped/skill-${id}/SKILL.md`,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  };
}

export class ResizeObserverMock {
  observe(): void {}
  disconnect(): void {}
  unobserve(): void {}
}

export function setupToolsPanelTest(): void {
  vi.stubGlobal('ResizeObserver', ResizeObserverMock);
  resetToolFilterState();
  toolFilterState.clear.mockReset();
  toolFilterState.clear.mockImplementation(() => {
    resetToolFilterState();
  });
  apiMock.listToolConfigs.mockResolvedValue([groupedTool, ungroupedTool]);
  apiMock.listToolGroups.mockResolvedValue([toolGroup]);
  apiMock.listUserspaceMountSources.mockResolvedValue([]);
  apiMock.getSettings.mockResolvedValue({
    settings: { show_tool_card_footer_actions: false },
  });
  apiMock.getToolHeartbeats.mockResolvedValue({ statuses: {} });
  apiMock.subscribeToolHealthEvents.mockReturnValue({
    addEventListener: vi.fn(),
    close: vi.fn(),
    onmessage: null,
  });
  apiMock.updateToolConfig.mockResolvedValue({ ...groupedTool, allow_write: true });
  apiMock.deleteToolConfig.mockResolvedValue(undefined);
  apiMock.createToolConfig.mockResolvedValue({
    ...ungroupedTool,
    id: 'tool-ungrouped-copy',
    name: 'Ungrouped Tool Copy',
    sort_order: 300,
  });
  apiMock.reorderTools.mockResolvedValue(undefined);
  apiMock.getPdmIndexStats.mockResolvedValue({
    document_count: 0,
    embedding_count: 0,
    last_indexed_at: null,
  });
  apiMock.getToolAccessPolicy.mockResolvedValue({
    tool_id: groupedTool.id,
    default_chat_access: 'deny',
    default_workspace_access: 'deny',
    users: [],
    groups: [],
  });
  apiMock.updateToolAccessPolicy.mockResolvedValue({
    tool_id: groupedTool.id,
    default_chat_access: 'read',
    default_workspace_access: 'deny',
    users: [],
    groups: [],
  });
  apiMock.listUsers.mockResolvedValue([]);
  apiMock.listAuthGroups.mockResolvedValue([]);
  apiMock.getInstructionSkillSummary.mockResolvedValue({ tools: {}, routes: {} });
  apiMock.copyToolInstructionSkills.mockResolvedValue([]);
  skillsDialogMock.nextSkills = [];
  contentProtectionMock.getConfig.mockResolvedValue(null);
}

export function teardownToolsPanelTest(): void {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
}
