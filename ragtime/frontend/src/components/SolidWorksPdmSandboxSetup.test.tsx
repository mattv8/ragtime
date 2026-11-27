import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SolidworksApiError, solidworksApi, type PdmSandboxResponse } from '../api/solidworks';
import {
  sandboxNameProblem,
  SolidWorksPdmSandboxSetup,
  type SandboxTarget,
  type SolidWorksPdmSandboxSetupProps,
} from './SolidWorksPdmSandboxSetup';

vi.mock('../api/solidworks', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api/solidworks')>()),
  solidworksApi: { createPdmSandbox: vi.fn() },
}));
vi.mock('./SolidWorksPdmFolderBrowser', () => ({
  SolidWorksPdmFolderBrowser: (props: {
    idPrefix?: string;
    disabled?: boolean;
    onPick?: (path: string) => void;
  }) => (
    <div
      data-testid="sandbox-browser"
      data-id-prefix={props.idPrefix}
      data-disabled={String(props.disabled === true)}
    >
      <input className="browser-filter-input" aria-label="Filter folders" />
      <button type="button" onClick={() => props.onPick?.('\\13_Ragtime')}>
        Pick 13_Ragtime
      </button>
    </div>
  ),
}));

const createSandbox = vi.mocked(solidworksApi.createPdmSandbox);
const TARGET: SandboxTarget = { toolId: 'sw-a', pdmToolId: 'pdm-a', vaultName: 'Ragtime' };
const FOLDER = '\\13_Ragtime\\Ragtime Sandbox';
const NO_DELETE =
  'Ragtime cannot delete vault folders. To remove one later, ask a PDM administrator.';
const CREATED: PdmSandboxResponse = {
  folder: FOLDER,
  covered: false,
  approved_folders: [FOLDER],
  allow_write: true,
  scripts_enabled: false,
};

function renderSandbox(overrides: Partial<SolidWorksPdmSandboxSetupProps> = {}) {
  const props: SolidWorksPdmSandboxSetupProps = {
    target: TARGET,
    unavailableReason: null,
    allowScripts: false,
    onBusyChange: vi.fn(),
    onCreated: vi.fn(),
    ...overrides,
  };
  render(<SolidWorksPdmSandboxSetup {...props} />);
  return props;
}

/** Renders with a target and opens the form on a picked parent, the usual starting point. */
function renderPicked(overrides: Partial<SolidWorksPdmSandboxSetupProps> = {}) {
  const props = renderSandbox(overrides);
  openForm();
  pickParent();
  return props;
}

function byId<T extends HTMLElement = HTMLElement>(id: string): T | null {
  return document.getElementById(id) as T | null;
}

function textOf(id: string): string {
  return byId(id)?.textContent ?? '';
}

function trigger() {
  return byId<HTMLButtonElement>('solidworks-pdm-sandbox-open') as HTMLButtonElement;
}

function createButton() {
  return byId<HTMLButtonElement>('solidworks-pdm-sandbox-create') as HTMLButtonElement;
}

function nameInput() {
  return byId<HTMLInputElement>('solidworks-pdm-sandbox-name') as HTMLInputElement;
}

function openForm() {
  fireEvent.click(trigger());
}

function pickParent() {
  fireEvent.click(screen.getByRole('button', { name: 'Pick 13_Ragtime' }));
}

function typeName(value: string) {
  fireEvent.change(nameInput(), { target: { value } });
}

function clickCreate() {
  fireEvent.click(createButton());
}

function stateOf() {
  return byId('solidworks-pdm-sandbox')?.dataset.swSandboxState;
}

/** Makes the next create reject, clicks Create and waits for the form-level alert. */
async function failCreate(error: unknown) {
  createSandbox.mockRejectedValueOnce(error);
  clickCreate();
  await waitFor(() => expect(byId('solidworks-pdm-sandbox-error')).not.toBeNull());
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe('SolidWorksPdmSandboxSetup', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      value: vi.fn(),
      configurable: true,
      writable: true,
    });
  });
  afterEach(() => {
    cleanup();
    document.body.innerHTML = '';
  });

  describe('idle and unavailable', () => {
    it('shows the group, the no-delete note and the closed trigger', () => {
      renderSandbox();
      const box = byId('solidworks-pdm-sandbox') as HTMLElement;
      expect(box.getAttribute('role')).toBe('group');
      expect(box.getAttribute('aria-labelledby')).toBe('solidworks-pdm-sandbox-title');
      expect(textOf('solidworks-pdm-sandbox-title')).toBe('Sandbox folder');
      expect(stateOf()).toBe('idle');
      expect(textOf('solidworks-pdm-sandbox-no-delete')).toBe(NO_DELETE);
      expect(trigger().textContent).toBe('Create sandbox folder…');
      expect(byId('solidworks-pdm-sandbox-form')).toBeNull();
    });

    it('disables the trigger and shows the reason when the target is unavailable', () => {
      renderSandbox({ target: null, unavailableReason: 'Enable the PDM tool first.' });
      expect(trigger().disabled).toBe(true);
      expect(textOf('solidworks-pdm-sandbox-reason')).toBe('Enable the PDM tool first.');
      expect(trigger().getAttribute('aria-describedby')).toBe('solidworks-pdm-sandbox-reason');
    });

    it('does not open when the target is unavailable', () => {
      renderSandbox({ target: null, unavailableReason: 'Save the connection first.' });
      openForm();
      expect(byId('solidworks-pdm-sandbox-form')).toBeNull();
    });
  });

  describe('form', () => {
    it('opens, hides the trigger, and focuses the parent browser filter', () => {
      renderSandbox();
      openForm();
      expect(trigger()).toBeNull();
      expect(byId('solidworks-pdm-sandbox-form')?.getAttribute('aria-label')).toBe(
        'Create sandbox folder',
      );
      expect(stateOf()).toBe('open');
      expect(document.activeElement).toBe(screen.getByLabelText('Filter folders'));
    });

    it('renders the parent browser with its own id prefix', () => {
      renderSandbox();
      openForm();
      const browser = screen.getByTestId('sandbox-browser');
      expect(browser.dataset.idPrefix).toBe('solidworks-pdm-sandbox-browser');
    });

    it('starts with no parent and blocks creation until one is chosen', () => {
      renderSandbox();
      openForm();
      expect(textOf('solidworks-pdm-sandbox-parent')).toBe('None chosen');
      expect(textOf('solidworks-pdm-sandbox-parent-help')).toBe('Choose a parent folder first.');
      expect(createButton().disabled).toBe(true);
    });

    it('shows the parent and the live target after a pick, and creates nothing', () => {
      const props = renderPicked();
      expect(textOf('solidworks-pdm-sandbox-parent')).toBe('\\13_Ragtime');
      expect(byId('solidworks-pdm-sandbox-parent-help')).toBeNull();
      expect(textOf('solidworks-pdm-sandbox-target')).toBe(
        'Will create \\13_Ragtime\\Ragtime Sandbox',
      );
      expect(createButton().disabled).toBe(false);
      expect(props.onCreated).not.toHaveBeenCalled();
    });

    it('keeps the default folder name and the name rules visible', () => {
      renderSandbox();
      openForm();
      expect(nameInput().value).toBe('Ragtime Sandbox');
      expect(nameInput().maxLength).toBe(64);
      expect(textOf('solidworks-pdm-sandbox-name-help')).toContain(
        'Letters, digits, spaces, dots, hyphens and underscores.',
      );
    });

    it('explains an invalid name and marks it invalid only after blur', () => {
      renderPicked();
      typeName('');
      expect(textOf('solidworks-pdm-sandbox-name-help')).toBe('Enter a folder name.');
      expect(nameInput().getAttribute('aria-invalid')).toBeNull();
      fireEvent.blur(nameInput());
      expect(nameInput().getAttribute('aria-invalid')).toBe('true');
      expect(byId('solidworks-pdm-sandbox-name-help')?.dataset.invalid).toBe('true');
      expect(createButton().disabled).toBe(true);
    });

    it('blocks a trailing dot or space and a leading symbol', () => {
      renderPicked();
      typeName('Sandbox.');
      expect(textOf('solidworks-pdm-sandbox-name-help')).toBe("Don't end with a dot or space.");
      typeName('_Sandbox');
      expect(textOf('solidworks-pdm-sandbox-name-help')).toBe('Start with a letter or digit.');
      expect(createButton().disabled).toBe(true);
    });

    it('enables creation for a valid name with a parent', () => {
      renderPicked();
      typeName('Lab 2.0-A_b');
      expect(createButton().disabled).toBe(false);
    });

    it('closes on Cancel and returns focus to the trigger', () => {
      renderPicked();
      fireEvent.click(byId('solidworks-pdm-sandbox-cancel') as HTMLElement);
      expect(byId('solidworks-pdm-sandbox-form')).toBeNull();
      expect(document.activeElement).toBe(trigger());
      expect(stateOf()).toBe('idle');
    });

    it('closes on Escape without reaching the wizard, and returns focus to the trigger', () => {
      const outside = vi.fn();
      document.addEventListener('keydown', outside);
      renderPicked();
      fireEvent.keyDown(nameInput(), { key: 'Escape' });
      expect(byId('solidworks-pdm-sandbox-form')).toBeNull();
      expect(document.activeElement).toBe(trigger());
      expect(outside).not.toHaveBeenCalled();
      document.removeEventListener('keydown', outside);
    });
  });

  describe('scripts line', () => {
    beforeEach(() => {
      const scripts = document.createElement('input');
      scripts.id = 'solidworks-allow-scripts';
      scripts.type = 'checkbox';
      document.body.appendChild(scripts);
    });

    it('says scripts are off and that this setup does not change it', () => {
      renderPicked({ allowScripts: false });
      expect(textOf('solidworks-pdm-sandbox-scripts')).toContain(
        "Allow scripts is off, so the AI can't run scripts. This setup doesn't change it.",
      );
    });

    it('says scripts are on and that they are not limited to the sandbox', () => {
      renderPicked({ allowScripts: true });
      expect(textOf('solidworks-pdm-sandbox-scripts')).toContain(
        "Allow scripts is on. Scripts run as this computer's Windows user, so they are not limited to the sandbox folder. This setup doesn't change it.",
      );
    });

    it('renders the link as a link, scoped by its id', () => {
      renderPicked();
      const link = byId('solidworks-pdm-sandbox-scripts-link') as HTMLButtonElement;
      expect(link.textContent).toBe('Go to Allow scripts');
      expect(link.classList.contains('sw-inline-link')).toBe(true);
    });

    it('focuses Allow scripts without changing its value', () => {
      renderPicked();
      const field = byId<HTMLInputElement>('solidworks-allow-scripts') as HTMLInputElement;
      field.checked = false;
      fireEvent.click(byId('solidworks-pdm-sandbox-scripts-link') as HTMLElement);
      expect(document.activeElement).toBe(field);
      expect(field.scrollIntoView).toHaveBeenCalledWith({ block: 'center' });
      expect(field.checked).toBe(false);
      expect(createSandbox).not.toHaveBeenCalled();
    });
  });

  describe('create', () => {
    it('sends the parent and the name, then reports success and the closed form', async () => {
      createSandbox.mockResolvedValue(CREATED);
      const props = renderPicked();
      clickCreate();
      await waitFor(() => expect(props.onCreated).toHaveBeenCalledWith(CREATED));
      expect(createSandbox).toHaveBeenCalledWith('sw-a', {
        pdm_tool_id: 'pdm-a',
        parent: '\\13_Ragtime',
        name: 'Ragtime Sandbox',
      });
      expect(byId('solidworks-pdm-sandbox-form')).toBeNull();
      expect(trigger()).toBeNull();
      expect(stateOf()).toBe('success');
      const success = byId('solidworks-pdm-sandbox-success') as HTMLElement;
      expect(success.textContent).toContain(
        'Created \\13_Ragtime\\Ragtime Sandbox. It is approved and Allow changes is on.',
      );
      expect(document.activeElement).toBe(success);
    });

    it('offers Create another, which reopens the form with the default name', async () => {
      createSandbox.mockResolvedValue(CREATED);
      renderPicked();
      typeName('Other');
      clickCreate();
      await waitFor(() => expect(byId('solidworks-pdm-sandbox-again')).not.toBeNull());
      fireEvent.click(byId('solidworks-pdm-sandbox-again') as HTMLElement);
      expect(stateOf()).toBe('open');
      expect(nameInput().value).toBe('Ragtime Sandbox');
      expect(byId('solidworks-pdm-sandbox-success')).toBeNull();
    });

    it('is busy while the request runs and reports busy to the wizard', async () => {
      const request = deferred<PdmSandboxResponse>();
      createSandbox.mockReturnValue(request.promise);
      const props = renderPicked();
      clickCreate();
      expect(stateOf()).toBe('busy');
      expect(props.onBusyChange).toHaveBeenLastCalledWith(true);
      expect(byId('solidworks-pdm-sandbox-form')?.getAttribute('aria-busy')).toBe('true');
      expect(nameInput().disabled).toBe(true);
      expect(createButton().textContent).toBe('Creating…');
      expect(textOf('solidworks-pdm-sandbox-result')).toContain(
        'Checking rights and creating the folder…',
      );
      expect(byId<HTMLButtonElement>('solidworks-pdm-sandbox-cancel')?.disabled).toBe(true);
      expect(screen.getByTestId('sandbox-browser').dataset.disabled).toBe('true');
      request.resolve(CREATED);
      await waitFor(() => expect(props.onBusyChange).toHaveBeenLastCalledWith(false));
    });

    it('ignores Escape while busy', () => {
      createSandbox.mockReturnValue(new Promise(() => undefined));
      renderPicked();
      clickCreate();
      fireEvent.keyDown(nameInput(), { key: 'Escape' });
      expect(byId('solidworks-pdm-sandbox-form')).not.toBeNull();
    });
  });

  describe('form-level errors', () => {
    it('lists the missing rights with a try-again action and keeps focus on Create', async () => {
      renderPicked();
      await failCreate(
        new SolidworksApiError('denied', 409, 'pdm_rights_missing', ['add_folders', 'check_out']),
      );
      expect(textOf('solidworks-pdm-sandbox-error')).toBe(
        'The PDM user is missing rights in \\13_Ragtime: add folders, check out. Grant them in the PDM Administration tool, or choose another parent.',
      );
      expect(byId('solidworks-pdm-sandbox-error')?.getAttribute('role')).toBe('alert');
      expect(document.activeElement).toBe(createButton());
      expect(textOf('solidworks-pdm-sandbox-retry')).toBe('Try again');
    });

    it('retries the create when Try again is chosen', async () => {
      const props = renderPicked();
      await failCreate(new SolidworksApiError('denied', 409, 'pdm_rights_missing', ['read']));
      createSandbox.mockResolvedValueOnce(CREATED);
      fireEvent.click(byId('solidworks-pdm-sandbox-retry') as HTMLElement);
      await waitFor(() => expect(props.onCreated).toHaveBeenCalledWith(CREATED));
      expect(createSandbox).toHaveBeenCalledTimes(2);
    });

    it.each([
      ['host_updating', 'warning', 'The SolidWorks agent is updating. Try again in a few minutes.'],
      [
        'host_not_running',
        'warning',
        "The SolidWorks agent isn't running. Start it in the agent panel above, then try again.",
      ],
    ])('explains %s as a %s', async (code, tone, text) => {
      renderPicked();
      await failCreate(new SolidworksApiError('x', 503, code));
      expect(textOf('solidworks-pdm-sandbox-error')).toBe(text);
      expect(byId('solidworks-pdm-sandbox-error')?.dataset.tone).toBe(tone);
      expect(byId('solidworks-pdm-sandbox-retry')).toBeNull();
    });

    it('shows a tool_disabled message as a form-level error', async () => {
      renderPicked();
      await failCreate(
        new SolidworksApiError(
          'Enable the SolidWorks tool and its PDM tool first.',
          409,
          'tool_disabled',
        ),
      );
      expect(textOf('solidworks-pdm-sandbox-error')).toBe(
        'Enable the SolidWorks tool and its PDM tool first.',
      );
      expect(nameInput().getAttribute('aria-invalid')).toBeNull();
      expect(stateOf()).toBe('open');
    });

    it('shows the overlap message from the server', async () => {
      const message =
        '\\13_Ragtime\\Ragtime Sandbox overlaps the approved folder \\13_Ragtime. Remove \\13_Ragtime or choose another name.';
      renderPicked();
      await failCreate(new SolidworksApiError(message, 409, 'sandbox_overlap'));
      expect(textOf('solidworks-pdm-sandbox-error')).toBe(message);
      expect(byId('solidworks-pdm-sandbox-error')?.dataset.tone).toBe('error');
    });

    it('shows the save-link message from the server', async () => {
      renderPicked();
      await failCreate(
        new SolidworksApiError(
          'Save the PDM link before creating a sandbox folder.',
          409,
          'pdm_link_not_saved',
        ),
      );
      expect(textOf('solidworks-pdm-sandbox-error')).toBe(
        'Save the PDM link before creating a sandbox folder.',
      );
    });

    it('uses the generic sentence for other codes', async () => {
      renderPicked();
      await failCreate(new SolidworksApiError('Folder changed meanwhile', 409, 'sandbox_conflict'));
      expect(textOf('solidworks-pdm-sandbox-error')).toBe(
        'Could not create the folder: Folder changed meanwhile',
      );
    });

    it('uses the generic sentence for a plain network error and stays open', async () => {
      renderPicked();
      await failCreate(new Error('offline'));
      expect(textOf('solidworks-pdm-sandbox-error')).toBe('Could not create the folder: offline');
      expect(stateOf()).toBe('open');
    });

    it('keeps the alert outside the polite status region', async () => {
      renderPicked();
      await failCreate(new SolidworksApiError('x', 409, 'pdm_error'));
      expect(
        byId('solidworks-pdm-sandbox-result')?.contains(byId('solidworks-pdm-sandbox-error')),
      ).toBe(false);
    });
  });

  describe('sandbox_exists field error', () => {
    const EXISTS =
      'A folder named “Ragtime Sandbox” already exists in \\13_Ragtime. Choose another name.';

    it('shows the message on the name input, focuses it, and keeps the parent', async () => {
      renderPicked();
      createSandbox.mockRejectedValueOnce(new SolidworksApiError(EXISTS, 409, 'sandbox_exists'));
      clickCreate();
      await waitFor(() => expect(textOf('solidworks-pdm-sandbox-name-help')).toBe(EXISTS));
      expect(nameInput().getAttribute('aria-invalid')).toBe('true');
      expect(byId('solidworks-pdm-sandbox-name-help')?.getAttribute('role')).toBe('alert');
      expect(document.activeElement).toBe(nameInput());
      expect(textOf('solidworks-pdm-sandbox-parent')).toBe('\\13_Ragtime');
      expect(byId('solidworks-pdm-sandbox-error')).toBeNull();
      expect(stateOf()).toBe('open');
    });

    it('clears the name error when the name changes and does not call onCreated', async () => {
      const props = renderPicked();
      createSandbox.mockRejectedValueOnce(new SolidworksApiError(EXISTS, 409, 'sandbox_exists'));
      clickCreate();
      await waitFor(() => expect(nameInput().getAttribute('aria-invalid')).toBe('true'));
      typeName('Other');
      expect(nameInput().getAttribute('aria-invalid')).toBeNull();
      expect(textOf('solidworks-pdm-sandbox-name-help')).toContain('Letters, digits');
      expect(props.onCreated).not.toHaveBeenCalled();
    });
  });

  describe('resets', () => {
    it('clears the failure when a new parent is picked', async () => {
      renderPicked();
      await failCreate(new SolidworksApiError('x', 409, 'pdm_error'));
      pickParent();
      expect(byId('solidworks-pdm-sandbox-error')).toBeNull();
    });

    it('clears the failure when the name changes', async () => {
      renderPicked();
      await failCreate(new SolidworksApiError('x', 409, 'pdm_error'));
      typeName('Other');
      expect(byId('solidworks-pdm-sandbox-error')).toBeNull();
    });

    it('drops the parent and errors when the target changes', async () => {
      const { rerender } = render(
        <SolidWorksPdmSandboxSetup
          target={TARGET}
          unavailableReason={null}
          allowScripts={false}
          onBusyChange={vi.fn()}
          onCreated={vi.fn()}
        />,
      );
      openForm();
      pickParent();
      await failCreate(new SolidworksApiError('x', 409, 'pdm_error'));
      rerender(
        <SolidWorksPdmSandboxSetup
          target={{ ...TARGET, vaultName: 'Other' }}
          unavailableReason={null}
          allowScripts={false}
          onBusyChange={vi.fn()}
          onCreated={vi.fn()}
        />,
      );
      expect(textOf('solidworks-pdm-sandbox-parent')).toBe('None chosen');
      expect(byId('solidworks-pdm-sandbox-error')).toBeNull();
    });
  });

  describe('sandbox folder name rule', () => {
    it.each([
      ['Ragtime Sandbox', null],
      ['a', null],
      ['0', null],
      ['', 'Enter a folder name.'],
      ['-lead', 'Start with a letter or digit.'],
      [
        'bad/name',
        'Use only letters, digits, spaces, dots, hyphens and underscores, up to 64 characters.',
      ],
      ['trail.', "Don't end with a dot or space."],
      ['trail ', "Don't end with a dot or space."],
    ])('checks %j', (name, problem) => {
      expect(sandboxNameProblem(name)).toBe(problem);
    });
  });

  describe('layout', () => {
    it('wraps the parent field in a group with its own id', () => {
      renderSandbox();
      openForm();
      const group = byId('solidworks-pdm-sandbox-parent-group') as HTMLElement;
      expect(group.contains(byId('solidworks-pdm-sandbox-parent'))).toBe(true);
      expect(group.contains(screen.getByTestId('sandbox-browser'))).toBe(true);
    });

    it('puts the result region above the Create and Cancel row', () => {
      renderPicked();
      const result = byId('solidworks-pdm-sandbox-result') as HTMLElement;
      const create = createButton();
      expect(
        result.compareDocumentPosition(create) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    });

    it('shows the overlap message as the server sent it', async () => {
      renderPicked();
      await failCreate(new SolidworksApiError('Overlap text.', 409, 'sandbox_overlap'));
      expect(textOf('solidworks-pdm-sandbox-error')).toBe('Overlap text.');
    });
  });
});
