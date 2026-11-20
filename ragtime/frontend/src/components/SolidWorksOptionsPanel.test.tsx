import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { SolidworksHostStatus, SolidworksOptions } from '../api/solidworks';
import { SolidWorksOptionsPanel, type SolidWorksOptionsPanelProps } from './SolidWorksOptionsPanel';

afterEach(cleanup);

function hostStatus(
  overrides: Partial<SolidworksHostStatus> & { versions?: number[] } = {},
): SolidworksHostStatus {
  const { versions = [2024, 2022], ...rest } = overrides;
  return {
    reachable: true,
    installed: true,
    running: true,
    host_version: '0.1.0',
    expected_host_version: '0.1.0',
    update_available: false,
    account: { name: 'HAMMERTON\\cad-agent', is_administrator: false },
    session: { interactive: true },
    solidworks: { installed_versions: versions, instances: [] },
    error: null,
    ...rest,
  };
}

function renderPanel(props: Partial<SolidWorksOptionsPanelProps> = {}) {
  const onChange = vi.fn();
  const onTimeoutMaxSecondsChange = vi.fn();
  const result = render(
    <SolidWorksOptionsPanel
      value={{}}
      onChange={onChange}
      hostStatus={null}
      timeoutMaxSeconds={100}
      onTimeoutMaxSecondsChange={onTimeoutMaxSecondsChange}
      {...props}
    />,
  );
  return { ...result, onChange, onTimeoutMaxSecondsChange };
}

const input = (id: string) => document.getElementById(id) as HTMLInputElement;

describe('SolidWorksOptionsPanel', () => {
  it('shows the defaults for background mode without the obsolete read-only note or write control', () => {
    renderPanel();
    expect(input('solidworks-max-instances').value).toBe('1');
    expect(input('solidworks-timeout-max-seconds').value).toBe('100');
    expect(input('solidworks-idle-minutes').value).toBe('15');
    expect(
      (document.getElementById('solidworks-instance-mode-background') as HTMLInputElement).checked,
    ).toBe(true);
    expect(document.getElementById('solidworks-read-only-note')).toBeNull();
    expect(screen.queryByRole('checkbox')).toBeNull();
    expect(screen.queryByText(/so there is no write setting/)).toBeNull();
  });

  it('uses the stable hook ids from the UX spec', () => {
    renderPanel();
    const panel = document.getElementById('solidworks-options-panel');
    expect(panel?.tagName).toBe('FIELDSET');
    expect(panel?.getAttribute('data-testid')).toBe('solidworks-options-panel');
    expect(document.getElementById('solidworks-instance-mode')?.getAttribute('role')).toBe(
      'radiogroup',
    );
    expect(document.getElementById('solidworks-timeout-range')).not.toBeNull();
    expect(document.getElementById('solidworks-timeout-help')).not.toBeNull();
    expect(document.getElementById('solidworks-max-instances-help')).not.toBeNull();
    expect(document.getElementById('solidworks-idle-minutes-help')).not.toBeNull();
    expect(document.getElementById('solidworks-version-help')).not.toBeNull();
  });

  it('switches mode, hides max instances and idle in attach mode, and keeps their values', () => {
    const { onChange, rerender } = renderPanel({
      value: { max_instances: 3, idle_shutdown_seconds: 1200 },
    });
    fireEvent.click(screen.getByLabelText(/Attach to open SolidWorks/));
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        instance_mode: 'attach',
        max_instances: 3,
        idle_shutdown_seconds: 1200,
      }),
    );
    rerender(
      <SolidWorksOptionsPanel
        value={{ instance_mode: 'attach', max_instances: 3, idle_shutdown_seconds: 1200 }}
        onChange={onChange}
        hostStatus={null}
        timeoutMaxSeconds={100}
        onTimeoutMaxSecondsChange={vi.fn()}
      />,
    );
    expect(document.getElementById('solidworks-max-instances')).toBeNull();
    expect(document.getElementById('solidworks-idle-minutes')).toBeNull();
    expect(input('solidworks-timeout-max-seconds').value).toBe('100');
    fireEvent.click(screen.getByLabelText(/Background instance/));
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        instance_mode: 'background',
        max_instances: 3,
        idle_shutdown_seconds: 1200,
      }),
    );
  });

  it('lists only Any installed version while the host status is unknown', () => {
    renderPanel({ hostStatus: null });
    const select = screen.getByLabelText('SolidWorks version') as HTMLSelectElement;
    expect([...select.options].map((option) => option.textContent)).toEqual([
      'Any installed version',
    ]);
    expect(document.getElementById('solidworks-version-warning')?.textContent).toContain(
      'Install and start the host',
    );
    expect(document.getElementById('solidworks-version-warning')?.getAttribute('data-tone')).toBe(
      'info',
    );
  });

  it('keeps a saved pin as "saved, not checked" while the status is unknown', () => {
    renderPanel({ value: { solidworks_version: 2022 }, hostStatus: null });
    const select = screen.getByLabelText('SolidWorks version') as HTMLSelectElement;
    expect(select.value).toBe('2022');
    expect(
      screen.getByRole('option', { name: 'SolidWorks 2022 (saved, not checked)' }),
    ).toBeTruthy();
  });

  it('lists the installed versions newest first when the host is running', () => {
    renderPanel({ hostStatus: hostStatus({ versions: [2022, 2024] }) });
    const select = screen.getByLabelText('SolidWorks version') as HTMLSelectElement;
    expect([...select.options].map((option) => option.textContent)).toEqual([
      'Any installed version',
      'SolidWorks 2024',
      'SolidWorks 2022',
    ]);
    expect(document.getElementById('solidworks-version-warning')).toBeNull();
  });

  it('keeps a pin that is not installed and warns, listing what was found', () => {
    renderPanel({
      value: { solidworks_version: 2019 },
      hostStatus: hostStatus({ versions: [2024, 2022] }),
    });
    const select = screen.getByLabelText('SolidWorks version') as HTMLSelectElement;
    expect(select.value).toBe('2019');
    expect(screen.getByRole('option', { name: 'SolidWorks 2019 (not installed)' })).toBeTruthy();
    const warning = document.getElementById('solidworks-version-warning');
    expect(warning?.getAttribute('data-tone')).toBe('warning');
    expect(warning?.textContent).toContain(
      'SolidWorks 2019 was not found on this machine (found: 2024, 2022)',
    );
    expect(select.getAttribute('aria-describedby')).toContain('solidworks-version-warning');
  });

  it('uses the empty-list warning when no SolidWorks is visible', () => {
    renderPanel({
      value: { solidworks_version: 2019 },
      hostStatus: hostStatus({ versions: [] }),
    });
    expect(document.getElementById('solidworks-version-warning')?.textContent).toBe(
      'SolidWorks 2019 was not found: no SolidWorks installation is visible to this account.',
    );
  });

  it('removes the stale not-installed option once the user changes away from it', () => {
    const onChange = vi.fn();
    const status = hostStatus({ versions: [2024] });
    const { rerender } = render(
      <SolidWorksOptionsPanel
        value={{ solidworks_version: 2019 }}
        onChange={onChange}
        hostStatus={status}
        timeoutMaxSeconds={100}
        onTimeoutMaxSecondsChange={vi.fn()}
      />,
    );
    fireEvent.change(screen.getByLabelText('SolidWorks version'), { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ solidworks_version: null }),
    );
    rerender(
      <SolidWorksOptionsPanel
        value={{ solidworks_version: null }}
        onChange={onChange}
        hostStatus={status}
        timeoutMaxSeconds={100}
        onTimeoutMaxSecondsChange={vi.fn()}
      />,
    );
    expect(screen.queryByRole('option', { name: 'SolidWorks 2019 (not installed)' })).toBeNull();
    expect(document.getElementById('solidworks-version-warning')).toBeNull();
  });

  it('describes the pin differently for background and attach', () => {
    const { rerender } = renderPanel({
      value: { solidworks_version: 2024, instance_mode: 'background' },
      hostStatus: hostStatus(),
    });
    expect(document.getElementById('solidworks-version-help')?.textContent).toBe(
      'The host only starts SolidWorks 2024. Calls fail if it is not installed.',
    );
    rerender(
      <SolidWorksOptionsPanel
        value={{ solidworks_version: 2024, instance_mode: 'attach' }}
        onChange={vi.fn()}
        hostStatus={hostStatus()}
        timeoutMaxSeconds={100}
        onTimeoutMaxSecondsChange={vi.fn()}
      />,
    );
    expect(document.getElementById('solidworks-version-help')?.textContent).toBe(
      'Calls fail if the open SolidWorks is not 2024.',
    );
  });

  it('shows the invalid max instances error and emits nothing until the draft is valid', () => {
    const { onChange } = renderPanel();
    const max = input('solidworks-max-instances');
    fireEvent.change(max, { target: { value: '9' } });
    expect(onChange).not.toHaveBeenCalled();
    expect(max.getAttribute('aria-invalid')).toBe('true');
    expect(max.getAttribute('aria-describedby')).toContain('solidworks-max-instances-error');
    const error = document.getElementById('solidworks-max-instances-error');
    expect(error?.textContent).toBe('Enter a whole number from 1 to 4.');
    expect(error?.getAttribute('data-tone')).toBe('error');
    fireEvent.change(max, { target: { value: '2' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ max_instances: 2 }));
    expect(document.getElementById('solidworks-max-instances-error')).toBeNull();
  });

  it('clamps an out-of-range max instances draft on blur', () => {
    const { onChange } = renderPanel({ value: { max_instances: 2 } });
    const max = input('solidworks-max-instances');
    fireEvent.change(max, { target: { value: '9' } });
    fireEvent.blur(max);
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ max_instances: 4 }));
    expect(max.value).toBe('4');
  });

  it('reverts an empty draft to the last valid value on blur', () => {
    const { onChange } = renderPanel({ value: { max_instances: 2 } });
    const max = input('solidworks-max-instances');
    fireEvent.change(max, { target: { value: '' } });
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.blur(max);
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ max_instances: 2 }));
    expect(max.value).toBe('2');
  });

  it('warns above 110 seconds of timeout without blocking the value', () => {
    renderPanel({ timeoutMaxSeconds: 120 });
    const warning = document.getElementById('solidworks-timeout-warning');
    expect(warning?.getAttribute('data-tone')).toBe('warning');
    expect(warning?.textContent).toContain('may be cut off by the web proxy');
    expect(input('solidworks-timeout-max-seconds').getAttribute('aria-describedby')).toContain(
      'solidworks-timeout-warning',
    );
  });

  it('shows no timeout warning at 110 seconds', () => {
    renderPanel({ timeoutMaxSeconds: 110 });
    expect(document.getElementById('solidworks-timeout-warning')).toBeNull();
  });

  it('shows an error under 15 seconds and does not emit it', () => {
    const { onTimeoutMaxSecondsChange } = renderPanel();
    const timeout = input('solidworks-timeout-max-seconds');
    fireEvent.change(timeout, { target: { value: '10' } });
    expect(onTimeoutMaxSecondsChange).not.toHaveBeenCalled();
    expect(document.getElementById('solidworks-timeout-error')?.textContent).toBe(
      'Enter at least 15 seconds.',
    );
    expect(timeout.getAttribute('aria-invalid')).toBe('true');
  });

  it('clamps the timeout to 15 and 3600 on blur', () => {
    const { onTimeoutMaxSecondsChange } = renderPanel();
    const timeout = input('solidworks-timeout-max-seconds');
    fireEvent.change(timeout, { target: { value: '5000' } });
    fireEvent.blur(timeout);
    expect(onTimeoutMaxSecondsChange).toHaveBeenLastCalledWith(3600);
    fireEvent.change(timeout, { target: { value: '1' } });
    fireEvent.blur(timeout);
    expect(onTimeoutMaxSecondsChange).toHaveBeenLastCalledWith(15);
  });

  it('keeps the timeout range element tied to the same value', () => {
    renderPanel({ timeoutMaxSeconds: 200 });
    const range = document.getElementById('solidworks-timeout-range') as HTMLInputElement;
    expect(range.type).toBe('range');
    expect(range.value).toBe('200');
  });

  it('converts idle minutes to seconds', () => {
    const { onChange } = renderPanel({ value: { idle_shutdown_seconds: 900 } });
    fireEvent.change(input('solidworks-idle-minutes'), { target: { value: '20' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ idle_shutdown_seconds: 1200 }),
    );
  });

  it('shows rounded minutes for a non-multiple of 60 without emitting until edited', () => {
    const { onChange } = renderPanel({ value: { idle_shutdown_seconds: 901 } });
    expect(input('solidworks-idle-minutes').value).toBe('15');
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.change(input('solidworks-idle-minutes'), { target: { value: '16' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ idle_shutdown_seconds: 960 }),
    );
  });

  it('rejects idle minutes outside 1 to 1440', () => {
    const { onChange } = renderPanel();
    fireEvent.change(input('solidworks-idle-minutes'), { target: { value: '0' } });
    expect(onChange).not.toHaveBeenCalled();
    expect(document.getElementById('solidworks-idle-minutes-error')?.textContent).toBe(
      'Enter a whole number from 1 to 1440.',
    );
  });

  it('disables the whole fieldset while disabled and keeps its help text', () => {
    renderPanel({ disabled: true });
    const panel = document.getElementById('solidworks-options-panel') as HTMLFieldSetElement;
    expect(panel.disabled).toBe(true);
    expect(document.getElementById('solidworks-max-instances-help')?.textContent).toContain(
      'How many SolidWorks processes',
    );
  });

  it('emits full options from the defaults when nothing was saved', () => {
    const { onChange } = renderPanel({ value: {} as SolidworksOptions });
    fireEvent.click(screen.getByLabelText(/Attach to open SolidWorks/));
    expect(onChange).toHaveBeenCalledWith({
      solidworks_version: null,
      instance_mode: 'attach',
      max_instances: 1,
      idle_shutdown_seconds: 900,
    });
  });
});
