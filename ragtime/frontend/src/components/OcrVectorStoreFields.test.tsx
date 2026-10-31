import { act, cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { OcrMode, OcrProvider } from '@/types';
import { isOcrTargetComplete, OcrVectorStoreFields } from './OcrVectorStoreFields';

const apiMock = vi.hoisted(() => ({
  getVisionModels: vi.fn(),
  getSettings: vi.fn(),
  getClaudeCodeAuthStatus: vi.fn(),
  getOpenAICodexAuthStatus: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function FieldsHarness({
  initialProvider = 'openai',
  onOpenOcrProviderSetup,
}: {
  initialProvider?: OcrProvider;
  onOpenOcrProviderSetup?: () => void;
}) {
  const [mode, setMode] = useState<OcrMode>('vision');
  const [provider, setProvider] = useState<OcrProvider | null>(initialProvider);
  const [model, setModel] = useState('gpt-4o');

  return (
    <OcrVectorStoreFields
      isLoading={false}
      ocrMode={mode}
      setOcrMode={setMode}
      ocrProvider={provider}
      setOcrProvider={setProvider}
      ocrVisionModel={model}
      setOcrVisionModel={setModel}
      vectorStoreType="faiss"
      setVectorStoreType={vi.fn()}
      onOpenOcrProviderSetup={onOpenOcrProviderSetup}
    />
  );
}

describe('OcrVectorStoreFields', () => {
  it('allows same-provider inheritance but requires a different provider model', () => {
    expect(isOcrTargetComplete('vision', 'openai', '', 'openai')).toBe(true);
    expect(isOcrTargetComplete('vision', 'anthropic', '', 'openai')).toBe(false);
    expect(isOcrTargetComplete('vision', 'anthropic', 'claude-sonnet', 'openai')).toBe(true);
  });

  it('offers API and subscription OCR providers and clears an overridden model', async () => {
    const user = userEvent.setup();
    render(<FieldsHarness />);

    const provider = screen.getByLabelText('Vision Provider');
    expect(screen.getByRole('option', { name: 'Anthropic API' })).toBeTruthy();
    expect(screen.getByRole('option', { name: 'Claude Code subscription' })).toBeTruthy();
    expect(screen.getByRole('option', { name: 'OpenAI Codex subscription' })).toBeTruthy();

    await user.selectOptions(provider, 'claude_code');
    expect((screen.getByLabelText('Vision Model') as HTMLSelectElement).value).toBe('');
  });

  it('loads provider metadata without sending credentials', async () => {
    const user = userEvent.setup();
    apiMock.getVisionModels.mockResolvedValue({
      success: true,
      models: [{ name: 'gpt-4o' }],
      message: '',
      provider: 'openai',
      base_url: '',
    });
    render(<FieldsHarness />);

    await user.click(screen.getByRole('button', { name: 'Load Models' }));

    expect(apiMock.getVisionModels).toHaveBeenCalledWith({ provider: 'openai' });
  });

  it('reports a missing Claude Code CLI and opens the caller setup action', async () => {
    const user = userEvent.setup();
    const openSetup = vi.fn();
    apiMock.getClaudeCodeAuthStatus.mockResolvedValue({
      installed: false,
      connected: false,
      has_oauth_token: false,
      has_cli_auth: false,
    });
    render(<FieldsHarness initialProvider="claude_code" onOpenOcrProviderSetup={openSetup} />);

    expect(await screen.findByText('Claude Code CLI is missing.')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Open provider setup' }));
    expect(openSetup).toHaveBeenCalledOnce();
  });

  it('shows the Claude Code status error for a disconnected CLI', async () => {
    apiMock.getClaudeCodeAuthStatus.mockResolvedValue({
      installed: true,
      connected: false,
      has_oauth_token: false,
      has_cli_auth: false,
      error: 'Run claude auth login to sign in.',
    });
    render(<FieldsHarness initialProvider="claude_code" />);

    expect(
      await screen.findByText(
        'Claude Code subscription is not configured. Run claude auth login to sign in.',
      ),
    ).toBeTruthy();
  });

  it('treats LLM provider settings as configured for local OCR but not embedding-only settings', async () => {
    apiMock.getSettings.mockResolvedValueOnce({
      settings: {
        llm_lmstudio_protocol: 'http',
        llm_lmstudio_host: 'localhost',
        llm_lmstudio_port: 1234,
      },
    });
    const { unmount } = render(<FieldsHarness initialProvider="lmstudio" />);

    expect(await screen.findByText('Saved LM Studio setup found.')).toBeTruthy();
    unmount();

    apiMock.getSettings.mockResolvedValueOnce({
      settings: {
        lmstudio_protocol: 'http',
        lmstudio_host: 'localhost',
        lmstudio_port: 1234,
      },
    });
    render(<FieldsHarness initialProvider="lmstudio" />);

    expect(await screen.findByText('LM Studio is not configured.')).toBeTruthy();
  });

  it('recognizes the saved Ollama LLM connection and its embedding fallback', async () => {
    apiMock.getSettings.mockResolvedValueOnce({
      settings: {
        llm_ollama_protocol: 'http',
        llm_ollama_host: 'ollama-llm',
        llm_ollama_port: 11434,
      },
    });
    const { unmount } = render(<FieldsHarness initialProvider="ollama" />);

    expect(await screen.findByText('Saved Ollama setup found.')).toBeTruthy();
    unmount();

    apiMock.getSettings.mockResolvedValueOnce({
      settings: {
        ollama_protocol: 'http',
        ollama_host: 'ollama-embedding',
        ollama_port: 11434,
      },
    });
    render(<FieldsHarness initialProvider="ollama" />);

    expect(await screen.findByText('Saved Ollama setup found.')).toBeTruthy();
  });

  it('opens the shared settings highlight in a new tab without a callback', async () => {
    const user = userEvent.setup();
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null);
    apiMock.getClaudeCodeAuthStatus.mockResolvedValue({
      installed: false,
      connected: false,
      has_oauth_token: false,
      has_cli_auth: false,
    });
    render(<FieldsHarness initialProvider="claude_code" />);

    await user.click(await screen.findByRole('button', { name: 'Open provider setup' }));

    expect(openSpy).toHaveBeenCalledWith(
      '?view=settings&highlight=llm_provider',
      '_blank',
      'noopener,noreferrer',
    );
    openSpy.mockRestore();
  });

  it('renders an alert when an actual provider status check fails', async () => {
    apiMock.getOpenAICodexAuthStatus.mockRejectedValue(new Error('status unavailable'));
    render(<FieldsHarness initialProvider="openai_codex" />);

    expect((await screen.findByRole('alert')).textContent).toContain(
      'Provider setup check failed: status unavailable',
    );
  });

  it('ignores stale model discovery after the provider changes', async () => {
    const user = userEvent.setup();
    let resolveOpenAi!: (value: unknown) => void;
    apiMock.getVisionModels.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveOpenAi = resolve;
      }),
    );
    render(<FieldsHarness />);

    await user.click(screen.getByRole('button', { name: 'Load Models' }));
    await user.selectOptions(screen.getByLabelText('Vision Provider'), 'claude_code');
    await act(async () => {
      resolveOpenAi({
        success: true,
        models: [{ name: 'gpt-4o' }],
        message: '',
        provider: 'openai',
        base_url: '',
      });
    });

    await vi.waitFor(() => expect(screen.queryByRole('option', { name: 'gpt-4o' })).toBeNull());
  });
});
