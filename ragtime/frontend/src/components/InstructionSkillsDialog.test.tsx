import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { InstructionSkillsDialog } from './InstructionSkillsDialog';
import type { BuiltinInstructionSkill } from '@/utils/instructionSkills';

const mocks = vi.hoisted(() => ({
  listInstructionSkills: vi.fn(),
  listBuiltinInstructionSkills: vi.fn(),
  createInstructionSkill: vi.fn(),
  updateInstructionSkill: vi.fn(),
  deleteInstructionSkill: vi.fn(),
  reorderInstructionSkills: vi.fn(),
}));
vi.mock('@/api', () => ({ api: mocks }));

const owner = { kind: 'tool' as const, id: 'tool-1' };
const skill = {
  id: 'one',
  owner_kind: 'tool' as const,
  owner_id: owner.id,
  name: 'first',
  description: 'First skill',
  content: 'Body',
  mode: 'always' as const,
  enabled: true,
  sort_order: 0,
  uri: 'skill://first',
  created_at: '',
  updated_at: '',
};

afterEach(cleanup);
beforeEach(() => {
  vi.resetAllMocks();
  mocks.listInstructionSkills.mockResolvedValue([skill]);
  mocks.listBuiltinInstructionSkills.mockResolvedValue([]);
});

const essentials = (overrides: Partial<BuiltinInstructionSkill> = {}): BuiltinInstructionSkill => ({
  name: 'solidworks-essentials',
  description: 'Read before any SolidWorks call.',
  mode: 'always',
  applies: true,
  reason: null,
  overridden: false,
  ...overrides,
});
const scripts = essentials({
  name: 'solidworks-scripts',
  description: 'Edit a copy with a script.',
  mode: 'on_demand',
  applies: false,
  reason: 'Needs Allow scripts',
});
const builtinRow = (name: string) =>
  document.querySelector<HTMLElement>(`[data-builtin-skill="${name}"]`);
const renderDialog = (skills: BuiltinInstructionSkill[] = [], onClose = vi.fn()) => {
  mocks.listBuiltinInstructionSkills.mockResolvedValue(skills);
  return render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={onClose} />);
};
const settle = () => act(() => new Promise((resolve) => setTimeout(resolve, 0)));

describe('InstructionSkillsDialog', () => {
  it('saves only changed editor fields and reports the persisted list', async () => {
    const changed = vi.fn();
    mocks.updateInstructionSkill.mockResolvedValue({ ...skill, description: 'Updated' });
    render(
      <InstructionSkillsDialog
        open
        owner={owner}
        ownerName="Tool"
        onClose={vi.fn()}
        onSkillsChanged={changed}
      />,
    );
    fireEvent.change(await screen.findByLabelText('Description'), { target: { value: 'Updated' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save skill' }));
    await waitFor(() =>
      expect(mocks.updateInstructionSkill).toHaveBeenCalledWith(owner, 'one', {
        description: 'Updated',
      }),
    );
    expect(changed).toHaveBeenCalledWith([{ ...skill, description: 'Updated' }]);
  });

  it('persists the enable toggle without changing editor fields', async () => {
    mocks.updateInstructionSkill.mockResolvedValue({ ...skill, enabled: false });
    const changed = vi.fn();
    render(
      <InstructionSkillsDialog
        open
        owner={owner}
        ownerName="Tool"
        onClose={vi.fn()}
        onSkillsChanged={changed}
      />,
    );
    fireEvent.click(await screen.findByLabelText('Enable skill first'));
    await waitFor(() =>
      expect(mocks.updateInstructionSkill).toHaveBeenCalledWith(owner, 'one', { enabled: false }),
    );
    expect(changed).toHaveBeenCalledWith([{ ...skill, enabled: false }]);
  });

  it('guards navigation when editing', async () => {
    const close = vi.fn();
    mocks.listInstructionSkills.mockResolvedValue([skill, { ...skill, id: 'two', name: 'second' }]);
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={close} />);
    fireEvent.change(await screen.findByLabelText('Description'), { target: { value: 'Changed' } });
    fireEvent.click(screen.getByRole('button', { name: /second/ }));
    expect(screen.getByRole('alertdialog')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Keep editing' }));
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(close).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Discard and close' }));
    expect(close).toHaveBeenCalledOnce();
  });

  it('creates a new skill draft', async () => {
    mocks.listInstructionSkills.mockResolvedValue([]);
    mocks.createInstructionSkill.mockResolvedValue(skill);
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole('button', { name: 'New skill' }));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'First' } });
    fireEvent.change(screen.getByLabelText('Description'), { target: { value: 'First skill' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save skill' }));
    await waitFor(() =>
      expect(mocks.createInstructionSkill).toHaveBeenCalledWith(owner, {
        name: 'first',
        description: 'First skill',
        content: '',
        mode: 'on_demand',
        enabled: true,
      }),
    );
  });

  it('reorders persisted rows with the keyboard', async () => {
    const second = { ...skill, id: 'two', name: 'second' };
    mocks.listInstructionSkills.mockResolvedValue([skill, second]);
    mocks.reorderInstructionSkills.mockResolvedValue([second, skill]);
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    fireEvent.keyDown(await screen.findByRole('button', { name: /second/ }), {
      key: 'ArrowUp',
      altKey: true,
    });
    await waitFor(() =>
      expect(mocks.reorderInstructionSkills).toHaveBeenCalledWith(owner, ['two', 'one']),
    );
  });

  it('flushes a pending reorder exactly once on unmount', async () => {
    const second = { ...skill, id: 'two', name: 'second' };
    mocks.listInstructionSkills.mockResolvedValue([skill, second]);
    mocks.reorderInstructionSkills.mockResolvedValue([second, skill]);
    const changed = vi.fn();
    const { unmount } = render(
      <InstructionSkillsDialog
        open
        owner={owner}
        ownerName="Tool"
        onClose={vi.fn()}
        onSkillsChanged={changed}
      />,
    );
    fireEvent.keyDown(await screen.findByRole('button', { name: /second/ }), {
      key: 'ArrowUp',
      altKey: true,
    });
    expect(mocks.reorderInstructionSkills).not.toHaveBeenCalled();
    unmount();
    expect(mocks.reorderInstructionSkills).toHaveBeenCalledTimes(1);
    expect(mocks.reorderInstructionSkills).toHaveBeenCalledWith(owner, ['two', 'one']);
    await waitFor(() => expect(changed).toHaveBeenCalledWith([second, skill]));
  });

  it('flushes a pending reorder immediately on close', async () => {
    const second = { ...skill, id: 'two', name: 'second' };
    mocks.listInstructionSkills.mockResolvedValue([skill, second]);
    mocks.reorderInstructionSkills.mockResolvedValue([second, skill]);
    const close = vi.fn();
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={close} />);
    fireEvent.keyDown(await screen.findByRole('button', { name: /second/ }), {
      key: 'ArrowUp',
      altKey: true,
    });
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(mocks.reorderInstructionSkills).toHaveBeenCalledWith(owner, ['two', 'one']);
    expect(close).toHaveBeenCalledOnce();
  });

  it('keeps a delayed reorder response scoped to its original owner', async () => {
    const second = { ...skill, id: 'two', name: 'second' };
    const other = { ...skill, id: 'other', owner_id: 'other-owner', name: 'other-skill' };
    let resolveReorder!: (items: (typeof skill)[]) => void;
    mocks.reorderInstructionSkills.mockReturnValue(
      new Promise((resolve) => {
        resolveReorder = resolve;
      }),
    );
    mocks.listInstructionSkills.mockImplementation(async (value: { id: string }) =>
      value.id === owner.id ? [skill, second] : [other],
    );
    const originalChanged = vi.fn();
    const otherChanged = vi.fn();
    const { rerender } = render(
      <InstructionSkillsDialog
        open
        owner={owner}
        ownerName="First"
        onClose={vi.fn()}
        onSkillsChanged={originalChanged}
      />,
    );
    fireEvent.keyDown(await screen.findByRole('button', { name: /second/ }), {
      key: 'ArrowUp',
      altKey: true,
    });
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    rerender(
      <InstructionSkillsDialog
        open={false}
        owner={owner}
        ownerName="First"
        onClose={vi.fn()}
        onSkillsChanged={originalChanged}
      />,
    );
    rerender(
      <InstructionSkillsDialog
        open
        owner={{ kind: 'tool', id: 'other-owner' }}
        ownerName="Other"
        onClose={vi.fn()}
        onSkillsChanged={otherChanged}
      />,
    );
    await screen.findByRole('button', { name: /other-skill/ });
    resolveReorder([second, skill]);
    await waitFor(() => expect(originalChanged).toHaveBeenCalledWith([second, skill]));
    expect(otherChanged).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /other-skill/ })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /second/ })).toBeNull();
  });

  it('clears dirty state after discarding the only draft', async () => {
    mocks.listInstructionSkills.mockResolvedValue([]);
    const close = vi.fn();
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={close} />);
    fireEvent.click(await screen.findByRole('button', { name: 'New skill' }));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'draft' } });
    fireEvent.click(screen.getByRole('button', { name: 'Discard draft' }));
    fireEvent.click(screen.getByRole('button', { name: 'Discard and continue' }));
    expect(screen.queryByText('Unsaved changes')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(close).toHaveBeenCalledOnce();
    expect(screen.queryByRole('alertdialog')).toBeNull();
  });

  it('shows duplicate-name errors from the server', async () => {
    mocks.updateInstructionSkill.mockRejectedValue(
      Object.assign(new Error('Conflict'), { status: 409 }),
    );
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    fireEvent.change(await screen.findByLabelText('Name'), { target: { value: 'renamed' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save skill' }));
    expect(await screen.findByText(/already exists on this tool/)).toBeTruthy();
  });

  it('keeps unsaved values when validation fails on the server', async () => {
    mocks.updateInstructionSkill.mockRejectedValue(
      Object.assign(new Error('Always-include budget exceeded'), { status: 422 }),
    );
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    fireEvent.change(await screen.findByLabelText('Description'), {
      target: { value: 'Changed' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save skill' }));
    expect((await screen.findByRole('alert')).textContent).toContain(
      'Always-include budget exceeded',
    );
    expect((screen.getByLabelText('Description') as HTMLTextAreaElement).value).toBe('Changed');
  });

  it('shows route-specific help and enabled always-skill budget', async () => {
    render(
      <InstructionSkillsDialog
        open
        owner={{ kind: 'route', id: 'route-1' }}
        ownerName="Route"
        onClose={vi.fn()}
      />,
    );
    expect(
      await screen.findByText('Sent to every agent that connects to this route.'),
    ).toBeTruthy();
    expect(screen.getByText(/4 of 32,000 characters/)).toBeTruthy();
  });

  it('shows default route empty-state guidance while custom routes retain their copy', async () => {
    mocks.listInstructionSkills.mockResolvedValue([]);
    const { rerender } = render(
      <InstructionSkillsDialog
        open
        owner={{ kind: 'route', id: 'default' }}
        ownerName="Default"
        onClose={vi.fn()}
      />,
    );
    expect(
      await screen.findByText(
        "Skills are instructions added to the default /mcp route, after Ragtime's built-in guide. Only clients that can use the general tool catalog receive them.",
      ),
    ).toBeTruthy();
    rerender(
      <InstructionSkillsDialog
        open
        owner={{ kind: 'route', id: 'route-1' }}
        ownerName="Route"
        onClose={vi.fn()}
      />,
    );
    expect(
      await screen.findByText(
        'Skills are instructions sent to every agent that connects to this route.',
      ),
    ).toBeTruthy();
  });

  it('shows default route always-mode guidance', async () => {
    render(
      <InstructionSkillsDialog
        open
        owner={{ kind: 'route', id: 'default' }}
        ownerName="Default"
        onClose={vi.fn()}
      />,
    );
    expect(
      await screen.findByText(
        "Sent to every eligible client on the default /mcp route. Workspace-only development credentials don't receive it.",
      ),
    ).toBeTruthy();
  });

  it('imports JSON-quoted frontmatter into a new draft', async () => {
    mocks.createInstructionSkill.mockResolvedValue({ ...skill, id: 'uploaded' });
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    await screen.findByLabelText('Description');
    const text = '---\nname: imported\ndescription: "A \\"quoted\\" description"\n---\nBody';
    const file = new File([text], 'skill.md', { type: 'text/markdown' });
    file.arrayBuffer = async () => new TextEncoder().encode(text).buffer;
    fireEvent.change(document.getElementById('instruction-skills-file-input')!, {
      target: { files: [file] },
    });
    await waitFor(() =>
      expect((screen.getByLabelText('Description') as HTMLTextAreaElement).value).toBe(
        'A "quoted" description',
      ),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Save skill' }));
    await waitFor(() =>
      expect(mocks.createInstructionSkill).toHaveBeenCalledWith(owner, {
        name: 'imported',
        description: 'A "quoted" description',
        content: 'Body',
        mode: 'on_demand',
        enabled: true,
      }),
    );
  });

  it('deletes a persisted skill after confirmation', async () => {
    mocks.deleteInstructionSkill.mockResolvedValue(undefined);
    const changed = vi.fn();
    render(
      <InstructionSkillsDialog
        open
        owner={owner}
        ownerName="Tool"
        onClose={vi.fn()}
        onSkillsChanged={changed}
      />,
    );
    fireEvent.click(await screen.findByRole('button', { name: 'Delete skill' }));
    await screen.findByRole('button', { name: 'Confirm?' }, { timeout: 4500 });
    fireEvent.click(screen.getByRole('button', { name: 'Confirm?' }));
    await waitFor(() => expect(mocks.deleteInstructionSkill).toHaveBeenCalledWith(owner, 'one'));
    expect(changed).toHaveBeenCalledWith([]);
  });
});

describe('InstructionSkillsDialog built-in skills', () => {
  it('shows no built-in section when the tool has none', async () => {
    renderDialog([]);
    await screen.findByLabelText('Description');
    await settle();
    expect(document.querySelector('[data-instruction-skills-builtin]')).toBeNull();
    expect(mocks.listBuiltinInstructionSkills).toHaveBeenCalledWith('tool-1');
  });

  it('keeps built-ins collapsed behind the toggle when admin skills exist', async () => {
    renderDialog([essentials(), scripts]);
    const toggle = await screen.findByRole('button', { name: /Built in \(2\)/ });
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('list', { name: 'Built-in skills' })).toBeNull();
    expect(toggle.textContent).toContain('1 active');
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(screen.getByRole('list', { name: 'Built-in skills' })).toBeTruthy();
    expect(
      screen.getByText(
        "Ship with Ragtime. You can't edit or turn them off. To change one, create a skill with the same name; yours replaces it.",
      ),
    ).toBeTruthy();
  });

  it('shows read-only rows with active and not-active status as text', async () => {
    renderDialog([essentials(), scripts]);
    fireEvent.click(await screen.findByRole('button', { name: /Built in \(2\)/ }));
    expect(builtinRow('solidworks-essentials')?.dataset.builtinState).toBe('active');
    expect(builtinRow('solidworks-essentials')?.textContent).toContain('Always');
    expect(builtinRow('solidworks-essentials')?.querySelector('input')).toBeNull();
    expect(builtinRow('solidworks-scripts')?.dataset.builtinState).toBe('inactive');
    expect(builtinRow('solidworks-scripts')?.textContent).toContain('Not active');
    expect(builtinRow('solidworks-scripts')?.textContent).toContain('Needs Allow scripts');
    expect(builtinRow('solidworks-scripts')?.textContent).toContain('On demand');
  });

  it('opens expanded when there are no admin skills', async () => {
    mocks.listInstructionSkills.mockResolvedValue([]);
    renderDialog([essentials()]);
    const list = await screen.findByRole('list', { name: 'Built-in skills' });
    expect(list).toBeTruthy();
    expect(
      screen.getByRole('button', { name: /Built in \(1\)/ }).getAttribute('aria-expanded'),
    ).toBe('true');
  });

  it('selects the admin skill that replaces a built-in', async () => {
    mocks.listInstructionSkills.mockResolvedValue([
      { ...skill, id: 'admin-1', name: 'solidworks-essentials' },
    ]);
    renderDialog([essentials()]);
    fireEvent.click(await screen.findByRole('button', { name: /Built in \(1\)/ }));
    expect(builtinRow('solidworks-essentials')?.dataset.builtinState).toBe('replaced');
    fireEvent.click(screen.getByRole('button', { name: 'Replaced by your skill' }));
    await waitFor(() =>
      expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe(
        'solidworks-essentials',
      ),
    );
    expect(screen.getByText('Replaces the built-in skill of the same name.')).toBeTruthy();
  });

  it('treats a disabled admin skill with the same name as replacing the built-in', async () => {
    mocks.listInstructionSkills.mockResolvedValue([
      { ...skill, id: 'admin-1', name: 'solidworks-essentials', enabled: false },
    ]);
    renderDialog([essentials()]);
    fireEvent.click(await screen.findByRole('button', { name: /Built in \(1\)/ }));
    expect(builtinRow('solidworks-essentials')?.dataset.builtinState).toBe('replaced');
    expect(builtinRow('solidworks-essentials')?.textContent).not.toContain('Active');
  });

  it('counts built-ins outside the budget with a note', async () => {
    renderDialog([essentials()]);
    await screen.findByRole('button', { name: /Built in \(1\)/ });
    expect(screen.getByText("Built-in skills aren't counted here.")).toBeTruthy();
  });

  it('hides the budget note when there are no built-ins', async () => {
    renderDialog([]);
    await screen.findByLabelText('Description');
    await settle();
    expect(document.getElementById('instruction-skills-budget-builtin-note')).toBeNull();
  });

  it('keeps the admin list usable when built-ins fail, and retries', async () => {
    mocks.listBuiltinInstructionSkills
      .mockRejectedValueOnce(new Error('down'))
      .mockResolvedValueOnce([essentials()]);
    renderDialog();
    expect(await screen.findByLabelText('Description')).toBeTruthy();
    await waitFor(() =>
      expect(document.getElementById('instruction-skills-builtin-error')).not.toBeNull(),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await screen.findByRole('button', { name: /Built in \(1\)/ });
    expect(mocks.listBuiltinInstructionSkills).toHaveBeenCalledTimes(2);
  });

  it('does not fetch built-ins for route owners', async () => {
    render(
      <InstructionSkillsDialog
        open
        owner={{ kind: 'route', id: 'default' }}
        ownerName="Default"
        onClose={vi.fn()}
      />,
    );
    await screen.findByLabelText('Description');
    expect(mocks.listBuiltinInstructionSkills).not.toHaveBeenCalled();
  });

  it('closes on Escape from the built-in toggle without a popover', async () => {
    const close = vi.fn();
    renderDialog([essentials()], close);
    fireEvent.keyDown(await screen.findByRole('button', { name: /Built in \(1\)/ }), {
      key: 'Escape',
    });
    expect(close).toHaveBeenCalledOnce();
  });
});

describe('InstructionSkillsDialog built-in toggle and status', () => {
  it('announces loading through the live region and disables the toggle', async () => {
    mocks.listBuiltinInstructionSkills.mockReturnValue(new Promise(() => {}));
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    await screen.findByLabelText('Description');
    await waitFor(() =>
      expect(document.getElementById('instruction-skills-builtin-status')?.textContent).toBe(
        'Loading built-in skills…',
      ),
    );
    const toggle = document.getElementById('instruction-skills-builtin-toggle');
    expect(toggle?.hasAttribute('disabled')).toBe(true);
    expect(toggle?.hasAttribute('aria-controls')).toBe(false);
  });

  it('disables the toggle and drops aria-controls when built-ins fail to load', async () => {
    mocks.listBuiltinInstructionSkills.mockRejectedValue(new Error('down'));
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    await waitFor(() =>
      expect(document.getElementById('instruction-skills-builtin-error')).not.toBeNull(),
    );
    const toggle = document.getElementById('instruction-skills-builtin-toggle');
    expect(toggle?.hasAttribute('disabled')).toBe(true);
    expect(toggle?.hasAttribute('aria-controls')).toBe(false);
    expect(document.getElementById('instruction-skills-builtin-status')?.textContent).toBe(
      'Could not load built-in skills.',
    );
  });

  it('points aria-controls at the list only while the list is rendered', async () => {
    renderDialog([essentials()]);
    const toggle = await screen.findByRole('button', { name: /Built in \(1\)/ });
    expect(toggle.hasAttribute('aria-controls')).toBe(false);
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-controls')).toBe('instruction-builtin-list');
    expect(document.getElementById('instruction-builtin-list')).not.toBeNull();
  });
});

describe('InstructionSkillsDialog built-in chevron', () => {
  it('shows no disclosure chevron on the toggle when there are no built-ins to open', async () => {
    mocks.listBuiltinInstructionSkills.mockRejectedValue(new Error('down'));
    render(<InstructionSkillsDialog open owner={owner} ownerName="Tool" onClose={vi.fn()} />);
    await waitFor(() =>
      expect(document.getElementById('instruction-skills-builtin-error')).not.toBeNull(),
    );
    expect(
      document.getElementById('instruction-skills-builtin-toggle')?.querySelector('svg'),
    ).toBeNull();
  });
});
