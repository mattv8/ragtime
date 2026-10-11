import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { detectJevKeyTransport } from '@/api/contentProtection';
import { JevClassifierFields } from './JevClassifierFields';
const props = {
  transport: 'auto' as const,
  model: 'jev-latest',
  onTransportChange: vi.fn(),
  onModelChange: vi.fn(),
  draftKey: '',
  onDraftKeyChange: vi.fn(),
  keyBusy: false,
  onSaveKey: vi.fn(),
  typesafeConfigured: false,
  openrouterConfigured: false,
  jevModels: [],
  jevModelsLoaded: false,
  jevModelsFetching: false,
  jevModelsError: null,
  onFetchModels: vi.fn(),
  readiness: null,
  readinessBusy: false,
  onCheckReadiness: vi.fn(),
  onOpenRouterLinkClick: vi.fn(),
};
describe('JevClassifierFields', () => {
  it('detects known Jev provider key prefixes without guessing unknown keys', () => {
    expect(detectJevKeyTransport(' ts_example')).toBe('typesafe');
    expect(detectJevKeyTransport('ts-example')).toBe('typesafe');
    expect(detectJevKeyTransport('sk-or-example')).toBe('openrouter');
    expect(detectJevKeyTransport('secret')).toBeNull();
  });

  it('announces an unrecognized key only after blur and disables saving and fetching in auto mode', async () => {
    const user = userEvent.setup();
    render(<JevClassifierFields {...props} draftKey="unknown" />);
    expect((screen.getByRole('button', { name: 'Save key' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect(
      (screen.getByRole('button', { name: 'Fetch Models' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(screen.queryByRole('alert')).toBeNull();
    await user.click(document.getElementById('jev-api-key-input')!);
    await user.click(screen.getByRole('textbox', { name: 'Model' }));
    expect(screen.getByRole('alert')).toBeTruthy();
  });
  it('allows an unrecognized key for TypeSafe but blocks it for OpenRouter', () => {
    const { rerender } = render(
      <JevClassifierFields {...props} transport="typesafe" draftKey="unknown" />,
    );
    expect((screen.getByRole('button', { name: 'Save key' }) as HTMLButtonElement).disabled).toBe(
      false,
    );
    rerender(<JevClassifierFields {...props} transport="openrouter" draftKey="unknown" />);
    expect((screen.getByRole('button', { name: 'Save key' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });
  it('saves a recognized OpenRouter draft as the shared key', async () => {
    const user = userEvent.setup();
    render(<JevClassifierFields {...props} draftKey="sk-or-example" />);
    await user.click(screen.getByRole('button', { name: 'Save key' }));
    expect(props.onSaveKey).toHaveBeenCalledWith('openrouter_api_key', 'sk-or-example');
    expect(document.getElementById('jev-openrouter-shared-warning')).toBeTruthy();
  });

  it('keeps readiness enabled while Jev-only controls are disabled for a generic backend', async () => {
    const user = userEvent.setup();
    render(<JevClassifierFields {...props} disabled draftKey="sk-or-example" />);
    const readinessButton = screen.getByRole('button', {
      name: 'Check readiness',
    }) as HTMLButtonElement;
    expect(readinessButton.disabled).toBe(false);
    expect((screen.getByRole('combobox') as HTMLSelectElement).disabled).toBe(true);
    expect((document.getElementById('jev-api-key-input') as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: 'Save key' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect(
      (screen.getByRole('button', { name: 'Fetch Models' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    await user.click(readinessButton);
    expect(props.onCheckReadiness).toHaveBeenCalledTimes(1);
  });

  it('disables readiness while a readiness check is busy', () => {
    render(<JevClassifierFields {...props} disabled={false} readinessBusy />);
    expect(
      (screen.getByRole('button', { name: /Check readiness/ }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
