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

    const name = screen.getByLabelText('Name');
    await user.clear(name);
    await user.tab();
    expect(onUpdate).not.toHaveBeenCalledWith('operational', { name: '' });
    expect(screen.getByRole('alert')).toBeTruthy();
    await user.click(name);
    await user.type(name, 'Changed');
    await user.keyboard('{Escape}');
    expect((name as HTMLInputElement).value).toBe('Operational');
  });

  it('leaves system categories read-only', () => {
    render(
      <ContentProtectionCategoryCard
        category={{ ...CATEGORY, system: true }}
        onUpdate={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect((screen.getByLabelText('Name') as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText('Description') as HTMLTextAreaElement).disabled).toBe(true);
    expect(screen.queryByLabelText('Threshold override')).toBeNull();
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

    expect((screen.getByLabelText('Name') as HTMLInputElement).disabled).toBe(false);
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
