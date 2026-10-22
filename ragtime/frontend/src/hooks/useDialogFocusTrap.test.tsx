import { describe, expect, it } from 'vitest';
import { getFocusableElements } from './useDialogFocusTrap';

describe('getFocusableElements', () => {
  it('excludes inactive tabs and controls hidden from the accessibility tree', () => {
    const container = document.createElement('div');
    container.innerHTML = `
      <button id="active">Active</button>
      <button id="inactive-tab" tabindex="-1">Inactive tab</button>
      <div hidden><button id="hidden">Hidden</button></div>
      <div inert><button id="inert">Inert</button></div>
      <div aria-hidden="true"><button id="aria-hidden">Aria hidden</button></div>
    `;

    expect(getFocusableElements(container).map((element) => element.id)).toEqual(['active']);
  });
});
