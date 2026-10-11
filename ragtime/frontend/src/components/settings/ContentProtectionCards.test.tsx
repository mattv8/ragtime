import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AccessLevel, ContentCategory } from '@/api/contentProtection';
import { ContentProtectionAccessLevelCard } from './ContentProtectionAccessLevelCard';
import { ContentProtectionCategoryCard } from './ContentProtectionCategoryCard';

const CATEGORY: ContentCategory = {
  id: 'operational',
  name: 'Operational',
  description: 'Operational information.',
  includes: [],
  excludes: [],
  examples: [],
  denial_message: 'Restricted.',
  threshold_override: 0.5,
  system: false,
};

afterEach(cleanup);

describe('ContentProtectionCategoryCard', () => {
  it('commits a typed threshold on blur', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <ContentProtectionCategoryCard category={CATEGORY} onUpdate={onUpdate} onDelete={vi.fn()} />,
    );

    const threshold = screen.getByLabelText('Threshold override');
    await user.clear(threshold);
    await user.type(threshold, '0.7');
    await user.tab();

    expect(onUpdate).toHaveBeenCalledWith('operational', { threshold_override: 0.7 });
  });

  it.each(['0', '1.1'])('keeps the committed threshold after invalid value %s', async (value) => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <ContentProtectionCategoryCard category={CATEGORY} onUpdate={onUpdate} onDelete={vi.fn()} />,
    );

    const threshold = screen.getByLabelText('Threshold override');
    await user.clear(threshold);
    await user.type(threshold, value);
    await user.tab();

    expect(onUpdate).not.toHaveBeenCalled();
    expect((threshold as HTMLInputElement).value).toBe('0.5');
    expect(screen.getByRole('alert')).toBeTruthy();
  });

  it('clears a threshold explicitly and cancels a draft with Escape', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <ContentProtectionCategoryCard category={CATEGORY} onUpdate={onUpdate} onDelete={vi.fn()} />,
    );

    const threshold = screen.getByLabelText('Threshold override');
    await user.clear(threshold);
    await user.tab();
    expect(onUpdate).toHaveBeenCalledWith('operational', { threshold_override: null });

    await user.type(threshold, '0.7');
    await user.keyboard('{Escape}');
    expect((threshold as HTMLInputElement).value).toBe('0.5');
  });

  it('preserves multiline text until blur and cancels required-field drafts', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <ContentProtectionCategoryCard category={CATEGORY} onUpdate={onUpdate} onDelete={vi.fn()} />,
    );

    const description = screen.getByLabelText('Description');
    await user.clear(description);
    await user.type(description, 'first line{enter}  second line  ');
    expect(onUpdate).not.toHaveBeenCalled();
    await user.tab();
    expect(onUpdate).toHaveBeenCalledWith('operational', {
      description: 'first line\n  second line  ',
    });

    await user.clear(description);
    await user.tab();
    expect(onUpdate).not.toHaveBeenCalledWith('operational', { description: '' });
    expect(screen.getByRole('alert')).toBeTruthy();
  });

  it('edits the name inline from the card heading', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <ContentProtectionCategoryCard category={CATEGORY} onUpdate={onUpdate} onDelete={vi.fn()} />,
    );

    expect(screen.queryByLabelText('Category name')).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Edit category name' }));
    const name = screen.getByLabelText('Category name');
    await user.clear(name);
    await user.type(name, 'Changed{Escape}');
    expect(onUpdate).not.toHaveBeenCalled();
    expect(screen.getByRole('heading', { name: 'Operational' })).toBeTruthy();

    await user.click(screen.getByRole('heading', { name: 'Operational' }));
    await user.clear(screen.getByLabelText('Category name'));
    await user.tab();
    expect(onUpdate).not.toHaveBeenCalled();

    await user.click(screen.getByRole('heading', { name: 'Operational' }));
    await user.clear(screen.getByLabelText('Category name'));
    await user.type(screen.getByLabelText('Category name'), '  Renamed  {Enter}');
    expect(onUpdate).toHaveBeenCalledWith('operational', { name: 'Renamed' });
  });

  it('leaves system categories read-only', () => {
    render(
      <ContentProtectionCategoryCard
        category={{ ...CATEGORY, system: true }}
        onUpdate={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Operational' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Edit category name' })).toBeNull();
    expect((screen.getByLabelText('Description') as HTMLTextAreaElement).disabled).toBe(true);
    expect(screen.queryByLabelText('Threshold override')).toBeNull();
  });

  it('hides the header while preserving category fields and delete state', () => {
    const { rerender } = render(
      <ContentProtectionCategoryCard
        category={CATEGORY}
        hideHeader
        onUpdate={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(document.querySelector('[data-category-id="operational"]')).toBeTruthy();
    expect(document.querySelector('.tool-card-header')).toBeNull();
    for (const label of [
      'Description',
      'Denial message',
      'Includes',
      'Excludes',
      'Examples',
      'Threshold override',
    ]) {
      expect(screen.getByLabelText(label)).toBeTruthy();
    }
    expect((screen.getByRole('button', { name: 'Delete' }) as HTMLButtonElement).disabled).toBe(
      false,
    );

    rerender(
      <ContentProtectionCategoryCard
        category={{ ...CATEGORY, system: true }}
        hideHeader
        onUpdate={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    for (const label of ['Description', 'Denial message', 'Includes', 'Excludes', 'Examples']) {
      expect(
        (screen.getByLabelText(label) as HTMLTextAreaElement | HTMLInputElement).disabled,
      ).toBe(true);
    }
    expect(screen.queryByLabelText('Threshold override')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Delete' })).toBeNull();
  });
});

describe('ContentProtectionAccessLevelCard', () => {
  it('associates each category checkbox with its description', () => {
    render(
      <ContentProtectionAccessLevelCard
        level={{ id: 'staff', name: 'Staff', granted_category_ids: [], guidance: '' }}
        categories={[CATEGORY]}
        onUpdate={vi.fn()}
      />,
    );

    const checkbox = screen.getByRole('checkbox', { name: 'Operational' });
    expect(checkbox.getAttribute('aria-describedby')).toBe(
      'content-protection-level-staff-category-operational-description',
    );
  });

  it('uses level-specific category description ids for cards in the same category', () => {
    render(
      <>
        <ContentProtectionAccessLevelCard
          level={{ id: 'staff', name: 'Staff', granted_category_ids: [], guidance: '' }}
          categories={[CATEGORY]}
          onUpdate={vi.fn()}
        />
        <ContentProtectionAccessLevelCard
          level={{ id: 'finance', name: 'Finance', granted_category_ids: [], guidance: '' }}
          categories={[CATEGORY]}
          onUpdate={vi.fn()}
        />
      </>,
    );

    expect(document.querySelectorAll('[id$="category-operational-description"]')).toHaveLength(2);
  });
});

describe('ContentProtectionAccessLevelCard', () => {
  it('keeps fields editable when only deletion is guarded', () => {
    const level: AccessLevel = {
      id: 'standard',
      name: 'Standard',
      granted_category_ids: [],
      guidance: '',
    };
    render(
      <ContentProtectionAccessLevelCard
        level={level}
        categories={[CATEGORY, { ...CATEGORY, id: 'rule_override', name: 'Rule override' }]}
        disabled
        idPrefix="settings-"
        onUpdate={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Standard' })).toBeTruthy();
    expect(screen.queryByLabelText('Rule override')).toBeNull();
    expect(document.querySelector('#settings-level-standard')).toBeTruthy();
  });

  it('explains why a disabled default or mapped level cannot be deleted', () => {
    const level: AccessLevel = {
      id: 'standard',
      name: 'Standard',
      granted_category_ids: [],
      guidance: '',
    };
    render(
      <ContentProtectionAccessLevelCard
        level={level}
        categories={[]}
        disabled
        deleteDisabledReason="This is the default access level."
        onUpdate={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect((screen.getByRole('button', { name: 'Delete' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect(screen.getByText('This is the default access level.')).toBeTruthy();
  });
});
