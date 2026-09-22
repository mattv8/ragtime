import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { UpdateSettingsRequest } from '@/types';
import { ChatSettingsSection } from './ChatSettingsSection';

vi.mock('@/api', () => ({
  api: { getOpenRouterCreditStatus: vi.fn().mockResolvedValue({ state: 'ok' }) },
}));

function renderSection(formData: UpdateSettingsRequest, isAdmin = true) {
  const setFormData = vi.fn();
  const handleSaveChat = vi.fn();
  render(
    <ChatSettingsSection
      open
      onToggle={vi.fn()}
      formData={formData}
      setFormData={setFormData}
      filteredChatModels={[]}
      manualDefaultChatModel={null}
      automaticDefaultChatModel={null}
      chatModelsLoading={false}
      toScopedModelIdentifier={(model) => `${model.provider}::${model.id}`}
      openModelFilterModal={vi.fn()}
      openOpenapiModelModal={vi.fn()}
      handleSaveChat={handleSaveChat}
      chatSaving={false}
      isAdmin={isAdmin}
      hasManagementApiKey={false}
    />,
  );
  return { setFormData, handleSaveChat };
}

describe('ChatSettingsSection', () => {
  it('puts the admin enable switch first and hides chat-only fields while disabled', () => {
    renderSection({ chat_enabled: false, default_chat_model: 'retained-model' });

    expect((document.getElementById('chat-enabled') as HTMLInputElement).checked).toBe(false);
    expect(
      screen.getByRole('switch', { name: 'Enable chat' }).getAttribute('aria-describedby'),
    ).toBe('chat-enabled-help');
    expect(screen.getByText('Configure Chat Models')).toBeTruthy();
    expect(screen.queryByText('Default Chat Model')).toBeNull();
    expect(screen.getByText('OpenAPI Models')).toBeTruthy();
    expect(screen.getByText('Cache Model Discovery')).toBeTruthy();
  });

  it('retains hidden draft values and sends only the toggle update through its save action', () => {
    const { setFormData, handleSaveChat } = renderSection({
      chat_enabled: false,
      default_chat_model: 'retained-model',
    });

    expect(setFormData).not.toHaveBeenCalled();
    expect(handleSaveChat).not.toHaveBeenCalled();
    const saveButtons = screen.getAllByRole('button', { name: 'Save Chat Settings' });
    fireEvent.click(saveButtons[saveButtons.length - 1]!);
    expect(handleSaveChat).toHaveBeenCalledTimes(1);
  });
});
