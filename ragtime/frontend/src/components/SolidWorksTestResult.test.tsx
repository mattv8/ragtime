import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { SolidWorksTestResult } from './SolidWorksTestResult';

afterEach(cleanup);

const successDetails = {
  host_version: '0.1.0',
  account: 'HAMMERTON\\cad-agent',
  solidworks_versions: [2024, 2022],
  session_interactive: true,
};

describe('SolidWorksTestResult', () => {
  it('renders nothing without a result', () => {
    const { container } = render(<SolidWorksTestResult result={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('shows the success facts with status role and passed pill', () => {
    render(
      <SolidWorksTestResult
        result={{
          success: true,
          message: 'SolidWorks host 0.1.0 is running as HAMMERTON\\cad-agent.',
          details: successDetails,
        }}
      />,
    );
    const panel = document.getElementById('solidworks-test-result');
    expect(panel?.getAttribute('role')).toBe('status');
    expect(panel?.getAttribute('data-sw-test-state')).toBe('success');
    expect(panel?.hasAttribute('data-sw-error-code')).toBe(false);
    expect(screen.getByText('Passed')).toBeTruthy();
    expect(
      screen.getByText('SolidWorks host 0.1.0 is running as HAMMERTON\\cad-agent.'),
    ).toBeTruthy();
    const facts = document.getElementById('solidworks-test-facts');
    expect(facts?.tagName).toBe('DL');
    expect(facts?.textContent).toContain('Host version0.1.0');
    expect(screen.getByText('2024, 2022')).toBeTruthy();
    expect(screen.getByText('Logged in')).toBeTruthy();
    expect(document.getElementById('solidworks-test-guidance')).toBeNull();
  });

  it('shows "None found" and "Not logged in" for empty or false facts', () => {
    render(
      <SolidWorksTestResult
        result={{
          success: true,
          message: 'ok',
          details: { solidworks_versions: [], session_interactive: false },
        }}
      />,
    );
    expect(screen.getByText('None found')).toBeTruthy();
    expect(screen.getByText('Not logged in')).toBeTruthy();
    expect(screen.getAllByText('Unknown').length).toBe(2);
  });

  it('uses its own pill with a tone attribute, not the shared status pill', () => {
    render(<SolidWorksTestResult result={{ success: false, message: 'Failed.' }} />);
    const pill = document.querySelector('#solidworks-test-result .sw-pill');
    expect(pill?.getAttribute('data-tone')).toBe('danger');
    expect(pill?.textContent).toBe('Failed');
    expect(document.querySelector('.userspace-status-pill')).toBeNull();
  });

  it('hides the guidance title when it repeats the message above it, keeping the hint', () => {
    render(
      <SolidWorksTestResult
        result={{
          success: false,
          message: 'The host did not answer.',
          details: { error_code: 'host_not_running' },
        }}
      />,
    );
    const guidance = document.getElementById('solidworks-test-guidance');
    expect(guidance?.querySelector('strong')).toBeNull();
    expect(guidance?.textContent).toContain('Install the host in the panel above');
  });

  it('shows the failure with alert role, guidance, code and the details disclosure', () => {
    render(
      <SolidWorksTestResult
        result={{
          success: false,
          message: 'SolidWorks is not open.',
          details: { error_code: 'solidworks_not_running', extra: 'debug' },
        }}
      />,
    );
    const panel = document.getElementById('solidworks-test-result');
    expect(panel?.getAttribute('role')).toBe('alert');
    expect(panel?.getAttribute('data-sw-test-state')).toBe('error');
    expect(panel?.getAttribute('data-sw-error-code')).toBe('solidworks_not_running');
    expect(screen.getByText('Failed')).toBeTruthy();
    expect(screen.getByText('SolidWorks is not open.', { selector: 'p' })).toBeTruthy();
    expect(document.getElementById('solidworks-test-guidance')?.textContent).toContain(
      'Attach mode needs SolidWorks already open',
    );
    expect(screen.getByText('solidworks_not_running').tagName).toBe('CODE');
    expect(screen.getByText('Show error details')).toBeTruthy();
    expect(document.querySelector('details pre')?.textContent).toContain('"extra": "debug"');
    expect(document.querySelector('.rc-command-block pre')).not.toBeNull();
  });

  it.each([
    ['ssh_unreachable', 'Ragtime could not sign in over SSH.'],
    ['ssh_auth_failed', 'Ragtime could not sign in over SSH.'],
    ['host_not_running', 'The host did not answer.'],
    ['protocol_mismatch', 'The host and Ragtime do not match.'],
    ['no_interactive_session', 'No desktop session.'],
    ['version_not_installed', 'That SolidWorks version is not installed.'],
    ['version_mismatch', 'The open SolidWorks is a different version.'],
    ['solidworks_not_running', 'SolidWorks is not open.'],
    ['solidworks_start_failed', 'SolidWorks did not start.'],
    ['instance_timeout', 'The call timed out.'],
    ['operation_timeout', 'The call timed out.'],
    ['instance_limit', 'All instances are busy.'],
  ])('shows guidance for %s', (code, title) => {
    render(
      <SolidWorksTestResult
        result={{ success: false, message: 'Failure', details: { error_code: code } }}
      />,
    );
    expect(document.getElementById('solidworks-test-guidance')?.textContent).toContain(title);
  });

  it('shows the message and code only for an unknown error code', () => {
    render(
      <SolidWorksTestResult
        result={{
          success: false,
          message: 'Something odd happened.',
          details: { error_code: 'other' },
        }}
      />,
    );
    expect(screen.getByText('Something odd happened.', { selector: 'p' })).toBeTruthy();
    expect(screen.getByText('other').tagName).toBe('CODE');
    expect(document.getElementById('solidworks-test-guidance')).toBeNull();
  });

  it('shows no guidance when the failure has no error code', () => {
    render(<SolidWorksTestResult result={{ success: false, message: 'Failed.' }} />);
    expect(document.getElementById('solidworks-test-guidance')).toBeNull();
    expect(document.querySelector('[data-sw-error-code]')).toBeNull();
  });
});
