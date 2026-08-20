import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ThemeChromeIcon } from './ThemeChromeIcon';

describe('ThemeChromeIcon', () => {
  it('renders the fallback and codicon variants inside one theme-aware wrapper', () => {
    const { container } = render(
      <ThemeChromeIcon
        fallback={<svg data-testid="fallback-icon" />}
        codicon="close"
        size={16}
        className="test-icon"
      />,
    );

    const wrapper = container.querySelector('.theme-chrome-icon');
    const fallback = screen.getByTestId('fallback-icon');
    const codicon = container.querySelector('.theme-chrome-icon-codicon');

    expect(wrapper).not.toBeNull();
    expect(wrapper?.className).toContain('test-icon');
    expect(wrapper?.getAttribute('aria-hidden')).toBe('true');
    expect(wrapper?.getAttribute('style')).toContain('--theme-chrome-icon-size: 16px;');
    expect(fallback.closest('.theme-chrome-icon-fallback')).not.toBeNull();
    expect(codicon).not.toBeNull();
    expect(codicon?.className).toContain('codicon');
    expect(codicon?.className).toContain('codicon-close');
    expect(codicon?.getAttribute('aria-hidden')).toBe('true');
  });
});
