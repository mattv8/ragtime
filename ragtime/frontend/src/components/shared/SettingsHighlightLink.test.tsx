import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SettingsHighlightLink } from './SettingsHighlightLink';

describe('SettingsHighlightLink', () => {
  afterEach(cleanup);
  it('renders a highlight URL and navigates plain clicks in-app', () => {
    const onNavigate = vi.fn();
    render(
      <SettingsHighlightLink settingId="content_protection" onNavigate={onNavigate}>
        Enable settings
      </SettingsHighlightLink>,
    );

    const link = screen.getByRole('link', { name: 'Enable settings' });
    expect(link.getAttribute('href')).toBe('?view=settings&highlight=content_protection');
    expect(fireEvent.click(link)).toBe(false);
    expect(onNavigate).toHaveBeenCalledWith('content_protection');
  });

  it('preserves modifier clicks and normal navigation without a callback', () => {
    const onNavigate = vi.fn();
    const { getByRole, rerender } = render(
      <SettingsHighlightLink settingId="content protection" onNavigate={onNavigate}>
        Enable settings
      </SettingsHighlightLink>,
    );

    const link = getByRole('link');
    expect(fireEvent.click(link, { ctrlKey: true })).toBe(true);
    expect(onNavigate).not.toHaveBeenCalled();

    rerender(
      <SettingsHighlightLink settingId="content protection">Enable settings</SettingsHighlightLink>,
    );
    expect(getByRole('link').getAttribute('href')).toBe(
      '?view=settings&highlight=content%20protection',
    );
  });
});
