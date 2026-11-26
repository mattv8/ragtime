import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ParsedSolidWorksResult } from '../utils/solidworksToolResult';
import { SolidWorksToolResult } from './SolidWorksToolResult';

afterEach(cleanup);

const base: ParsedSolidWorksResult = {
  state: 'ok',
  operation: 'status',
  title: 'Status',
  args: [],
  elapsedMs: 420,
  solidworksVersion: '2024',
  truncated: null,
  error: null,
  result: null,
};

function show(
  overrides: Partial<ParsedSolidWorksResult>,
  options: { artifactShown?: boolean; rawOutput?: string } = {},
) {
  const view = render(
    <SolidWorksToolResult
      parsed={{ ...base, ...overrides }}
      rawOutput={options.rawOutput ?? '{}'}
      resultId="result"
      artifactShown={options.artifactShown ?? false}
      renderMarkdown={(markdown) => <div>{markdown}</div>}
    />,
  );
  expect(view.container.querySelector('pre')).toBeNull();
  return view;
}

describe('SolidWorksToolResult', () => {
  it('keeps argument and fact label hooks', () => {
    const view = show({
      args: [{ key: 'limit', label: 'Limit', value: '20', kind: 'number' }],
      result: { host_version: '1.2' },
    });
    expect(view.container.querySelector('.solidworks-result-args dt')?.textContent).toBe('Limit');
    expect(view.container.querySelector('.solidworks-result-facts dt')?.textContent).toBe(
      'Agent version',
    );
  });

  it('uses the SolidWorks label and hides zero elapsed time', () => {
    show({ title: 'Preview model', elapsedMs: 0 });
    expect(screen.getByText('SolidWorks')).toBeDefined();
    expect(screen.queryByText('Preview model')).toBeNull();
    expect(screen.queryByText('0 ms')).toBeNull();
  });

  it('formats artifact format and size like the artifact card', () => {
    show({
      operation: 'model.preview',
      result: { artifact: { file_name: 'model.png', format: 'png', size_bytes: 31499 } },
    });
    expect(screen.getByText('PNG')).toBeDefined();
    expect(screen.getByText('30.8 KB')).toBeDefined();
  });

  it('shows requested guide markdown before secondary topics', () => {
    const view = show({
      operation: 'guide',
      args: [{ key: 'topic', label: 'Topic', value: 'previews', kind: 'text' }],
      result: {
        topic: 'previews',
        markdown: 'Requested guide',
        topics: [
          { id: 'previews', title: 'Previews' },
          { id: 'overview', title: 'Overview and first steps' },
        ],
      },
    });
    const guide = view.container.querySelector('[data-solidworks-guide]')!;
    expect(guide.firstElementChild?.textContent).toContain('Requested guide');
    expect(guide.querySelector('[data-solidworks-other-topics] code')?.getAttribute('title')).toBe(
      'Overview and first steps',
    );
    expect(guide.querySelector('ul')).toBeNull();
  });

  it('shows the topic list before default guide markdown', () => {
    const view = show({
      operation: 'guide',
      result: {
        topic: 'overview',
        markdown: 'Default guide',
        topics: [{ id: 'overview', title: 'Overview and first steps', when_to_read: 'Read first' }],
      },
    });
    const guide = view.container.querySelector('[data-solidworks-guide]')!;
    expect(guide.firstElementChild?.tagName).toBe('UL');
    expect(guide.querySelector('ul.solidworks-result-topics li strong')?.textContent).toBe(
      'Overview and first steps',
    );
    expect(guide.querySelector('li .solidworks-result-topic-description')?.textContent).toBe(
      'Read first',
    );
    expect(guide.lastElementChild?.textContent).toContain('Default guide');
    expect(screen.getByText('Overview')).toBeDefined();
  });

  it('groups undotted operations without repeating their names as headings', () => {
    const view = show({
      operation: 'describe',
      result: {
        operations: [
          { name: 'describe', summary: 'Describe operations', access: 'read' },
          { name: 'status', summary: 'Show status', access: 'read' },
          { name: 'model.export', summary: 'Export', access: 'write', paged: true },
        ],
      },
    });
    const groups = view.container.querySelectorAll('[data-operation-group]');
    expect(groups[0].getAttribute('data-operation-group')).toBe('');
    expect(groups[0].querySelector('strong')).toBeNull();
    expect(groups[1].querySelector('strong')?.textContent).toBe('model');
    expect(groups[1].querySelector('.solidworks-result-operation-summary')?.textContent).toBe(
      'Export',
    );
  });

  it('humanises document types and keeps the full path in a title', () => {
    show({
      operation: 'documents.list',
      result: { items: [{ title: 'Part', type: 'assembly', path: 'C:\\long\\part.sldasm' }] },
    });
    expect(screen.getByText('Assembly')).toBeDefined();
    expect(screen.getByTitle('C:\\long\\part.sldasm').className).toBe('solidworks-result-path');
  });

  it.each([
    ['status', { host_version: '1.2' }, '1.2'],
    [
      'describe',
      { operations: [{ name: 'model.export', summary: 'Export', access: 'write' }] },
      'Export',
    ],
    ['guide', { markdown: '# Guide' }, '# Guide'],
    ['documents.list', { documents: [{ title: 'Part', type: 'Part' }] }, 'Part'],
    ['components.list', { components: [{ name: 'Wheel' }] }, 'Wheel'],
    ['document.open', { title: 'Assembly' }, 'Assembly'],
    ['model.export', { file_name: 'part.step', format: 'STEP' }, 'part.step'],
    ['unknown', { nested: { value: 'hello' } }, 'hello'],
  ])('renders %s', (operation, result, expected) => {
    show({ operation, result });
    expect(screen.getAllByText(expected).length).toBeGreaterThan(0);
  });

  it('renders error without success', () => {
    show({ state: 'error', error: { code: 'bad', message: 'Invalid', details: [] } });
    expect(screen.getByRole('group', { name: 'Error' }).textContent).toContain('Invalid');
    expect(screen.queryByText('Completed')).toBeNull();
  });

  it('renders truncation', () => {
    show({ truncated: 'marker' });
    expect(screen.getByText(/Output was truncated in chat history/)).toBeDefined();
  });

  it('renders unparsed output', () => {
    show({ state: 'unparsed' });
    expect(screen.getByText(/Unable to parse/)).toBeDefined();
    expect(screen.getByRole('button', { name: 'Copy JSON' })).toBeDefined();
  });

  it('renders running state', () => {
    show({ state: 'running' });
    expect(screen.getByRole('status').textContent).toContain('Waiting for SolidWorks');
  });

  it('renders status facts, instances and PDM views', () => {
    show({
      result: {
        host_version: '1.2',
        protocol: 1,
        account: { name: 'alice', is_administrator: true },
        session: { interactive: true },
        uptime_seconds: 7440,
        solidworks: {
          installed_versions: ['2025'],
          instances: [
            {
              id: 'i1',
              mode: 'background',
              version: '2025',
              revision: '33',
              documents: 2,
              busy: true,
            },
          ],
        },
        pdm: { views: [{ vault: 'Vault', root: 'C:\\Vault', logged_in: true, user: 'alice' }] },
      },
    });
    for (const value of [
      'Administrator',
      'Interactive',
      '2 h 4 min',
      '2025',
      'i1',
      'background',
      'Busy',
      'Vault',
      'C:\\Vault',
      'Logged in',
    ]) {
      expect(screen.getAllByText(value).length).toBeGreaterThan(0);
    }
    expect(screen.getAllByRole('table')).toHaveLength(2);
  });

  it('copies a document path with feedback', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
    show({ operation: 'document.close', result: { path: 'C:\\part.sldprt' } });
    fireEvent.click(screen.getByRole('button', { name: 'Copy path' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Copied path' })).toBeDefined());
    expect(writeText).toHaveBeenCalledWith('C:\\part.sldprt');
    vi.unstubAllGlobals();
  });

  it('shows omission notes in guide topics and markdown', () => {
    show({
      operation: 'guide',
      result: {
        topics: [
          {
            id: 'topic',
            title: 'Before ... [123 characters omitted] ... after',
            when_to_read: '... [9 characters omitted] ...',
          },
        ],
        markdown: 'Before ... [4,896 characters omitted] ... after',
      },
    });
    expect(screen.getAllByRole('note')).toHaveLength(3);
    expect(screen.getByText('123 characters omitted')).toBeDefined();
    expect(screen.getByText('4,896 characters omitted')).toBeDefined();
  });

  it.each([true, false])('advertises only a rendered artifact: %s', (visible) => {
    show({ operation: 'model.preview', result: { artifact: {} } }, { artifactShown: visible });
    expect(Boolean(screen.queryByText('Shown in the card below'))).toBe(visible);
  });

  it('uses defined badge tones for success, neutral, and warning', () => {
    show({
      operation: 'describe',
      result: {
        operations: [
          { name: 'a', access: 'write' },
          { name: 'b', access: 'read' },
        ],
      },
    });
    expect(screen.getByText('Completed').className).toBe('badge completed');
    expect(screen.getByText('write').className).toBe('badge solidworks-result-badge-warning');
    expect(screen.getByText('read').className).toBe('badge badge-muted');
  });

  it('shows error badges and a fallback for an absent error', () => {
    const view = show({ state: 'error', error: null });
    expect(screen.queryByText('Failed')).toBeNull();
    expect(screen.getByText('SolidWorks reported a failure.')).toBeDefined();
    view.unmount();
    show({ state: 'error', error: { code: 'bad', message: 'Invalid', details: [] } });
    expect(screen.getByText('bad').className).toBe('badge failed');
    expect(screen.queryByText('Failed')).toBeNull();
  });

  it('distinguishes empty output from malformed output', () => {
    show({ state: 'unparsed' }, { rawOutput: '' });
    expect(screen.getByText('No output was recorded.')).toBeDefined();
  });

  it('labels host and agent independently', () => {
    show({ operation: 'document.open', result: { opened_by_host: true, opened_by_agent: true } });
    expect(screen.getByText('Opened by host')).toBeDefined();
    expect(screen.getByText('Opened by agent')).toBeDefined();
  });

  it('renders duplicate rows without React key warnings', () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const view = show({
      operation: 'components.list',
      result: { items: [{ name: 'same' }, { name: 'same' }] },
    });
    expect(screen.getAllByText('same')).toHaveLength(2);
    view.unmount();
    show({
      result: {
        solidworks: { instances: [{ id: 'same' }, { id: 'same' }] },
        pdm: { views: [{ vault: 'v' }, { vault: 'v' }] },
      },
    });
    expect(errors).not.toHaveBeenCalled();
    errors.mockRestore();
  });

  it('uses the injected markdown renderer', () => {
    const renderMarkdown = vi.fn((value: string) => <div data-testid="markdown">{value}</div>);
    render(
      <SolidWorksToolResult
        parsed={{ ...base, operation: 'guide', result: { markdown: 'Guide' } }}
        rawOutput="{}"
        resultId="guide"
        artifactShown={false}
        renderMarkdown={renderMarkdown}
      />,
    );
    expect(renderMarkdown).toHaveBeenCalledWith('Guide');
    expect(screen.getByTestId('markdown').textContent).toBe('Guide');
  });
});

function headersOf(root: ParentNode | null | undefined): (string | null)[] {
  return Array.from(root?.querySelectorAll('th') ?? []).map((th) => th.textContent);
}

function factValue(root: ParentNode, term: string): string | undefined {
  const dt = Array.from(root.querySelectorAll('dt')).find((item) => item.textContent === term);
  return dt?.nextElementSibling?.textContent ?? undefined;
}

describe('SolidWorks operation fields', () => {
  it('shows a copied model with its folder, file count and warnings as text', () => {
    const view = show({
      operation: 'document.copy',
      result: {
        folder: 'C:\\work\\copies\\a1',
        path: 'C:\\work\\copies\\a1\\CU.SLDASM',
        count: 12,
        files: [],
        warnings: ['Missing source: C:\\PDM\\Bracket.SLDPRT'],
      },
    });
    const copy = view.container.querySelector('[data-solidworks-copy]');
    expect(factValue(copy as Element, 'Folder')).toBe('C:\\work\\copies\\a1');
    expect(factValue(copy as Element, 'Model')).toBe('C:\\work\\copies\\a1\\CU.SLDASM');
    expect(factValue(copy as Element, 'Files')).toBe('12');
    expect(Array.from(copy?.querySelectorAll('code') ?? []).map((code) => code.title)).toContain(
      'C:\\work\\copies\\a1',
    );
    expect(screen.getByText('Missing source: C:\\PDM\\Bracket.SLDPRT')).toBeDefined();
  });

  it("stacks a mate's error, what is wrong and entities under its name", () => {
    const view = show({
      operation: 'mates.list',
      result: {
        items: [
          {
            name: 'Coincident1',
            type: 'coincident',
            suppressed: false,
            error: {
              code: '7',
              name: 'swMateErrorBroken',
              meaning: 'The mate references a missing part.',
            },
            whats_wrong: 'Reference missing',
            entities: [
              {
                component: 'Part1-1',
                component_path: 'Sub-1/Part1-1',
                entity_type: 'face',
                resolved: false,
              },
            ],
          },
          {
            name: 'Distance1',
            type: 'distance',
            suppressed: false,
            error: null,
            whats_wrong: null,
            entities: [],
          },
        ],
        total: 2,
        offset: 0,
      },
    });
    const list = view.container.querySelector('[data-solidworks-list="mates.list"]');
    expect(headersOf(list)).toEqual(['Name', 'Type', 'Suppressed']);
    const rows = Array.from(list?.querySelectorAll('tbody tr') ?? []);
    const diagnosis = rows[0].querySelector('[data-solidworks-mate-diagnosis]');
    expect(diagnosis?.textContent).toContain('swMateErrorBroken');
    expect(diagnosis?.textContent).toContain('The mate references a missing part.');
    expect(diagnosis?.textContent).toContain("What's wrong: Reference missing");
    expect(diagnosis?.textContent).toContain('Sub-1/Part1-1');
    expect(diagnosis?.textContent).toContain('Unresolved');
    expect(rows[1].querySelector('[data-solidworks-mate-diagnosis]')).toBeNull();
  });

  it('labels missing files and lightweight components', () => {
    show({
      operation: 'components.list',
      result: {
        items: [
          {
            name: 'Bracket-1',
            configuration: 'Default',
            file_exists: false,
            suppression: 'missing_file',
            children: 0,
          },
          {
            name: 'Plate-1',
            configuration: 'Default',
            file_exists: true,
            suppression: 'lightweight',
            children: 0,
          },
        ],
        total: 2,
        offset: 0,
      },
    });
    expect(screen.getByText('Missing file').closest('td')?.textContent).toContain('Bracket-1');
    expect(screen.getByText('Lightweight').closest('td')?.textContent).toContain('Plate-1');
  });

  it('shows load warnings and missing reference paths from an opened document', () => {
    const view = show({
      operation: 'document.open',
      result: {
        title: 'CU-IDB0061',
        type: 'assembly',
        document_id: 'doc-1',
        path: 'C:\\PDM\\CU-IDB0061.SLDASM',
        load_warnings: ['references_not_found'],
        missing_references: ['C:\\HAM-PDM\\Bracket.SLDPRT', 'C:\\HAM-PDM\\Plate.SLDPRT'],
      },
    });
    expect(screen.getByText('References not found')).toBeDefined();
    expect(screen.getByText('Missing references (2)')).toBeDefined();
    expect(
      view.container.querySelector(
        '[data-solidworks-missing-references] code[title$="Plate.SLDPRT"]',
      ),
    ).not.toBeNull();
  });

  it('omits the missing-reference block when every reference resolves', () => {
    const view = show({
      operation: 'document.open',
      result: { title: 'Part', document_id: 'doc-2', load_warnings: [], missing_references: [] },
    });
    expect(view.container.querySelector('[data-solidworks-missing-references]')).toBeNull();
  });
});

describe('SolidWorks results from older agents', () => {
  it('keeps the mates table of agent 0.5.0, which omits the error and entity fields', () => {
    const view = show({
      operation: 'mates.list',
      result: {
        items: [
          {
            name: 'Coincident1',
            type: 'coincident',
            suppressed: false,
            error_code: '0',
            warning: false,
            entities: [{ component: 'Part1-1', entity_type: 'face' }],
          },
        ],
        total: 1,
        offset: 0,
      },
    });
    const headers = Array.from(view.container.querySelectorAll('th')).map((th) => th.textContent);
    expect(headers).toEqual(['Name', 'Type', 'Suppressed']);
    expect(view.container.textContent).not.toContain('undefined');
    expect(view.container.textContent).not.toContain('Unresolved');
    expect(view.container.querySelector('[data-solidworks-mate-diagnosis]')).toBeNull();
  });

  it('shows no missing-file or lightweight marks for components without the new fields', () => {
    const view = show({
      operation: 'components.list',
      result: {
        items: [{ name: 'Bracket-1', configuration: 'Default', suppressed: false, children: 0 }],
        total: 1,
        offset: 0,
      },
    });
    expect(view.container.textContent).not.toContain('undefined');
    expect(view.container.textContent).not.toContain('Missing file');
    expect(view.container.textContent).not.toContain('Lightweight');
  });

  it('renders an opened document without load warnings or missing references', () => {
    const view = show({
      operation: 'document.open',
      result: { title: 'Part1', type: 'part', document_id: 'doc-1', path: 'C:\\Part1.SLDPRT' },
    });
    expect(view.container.textContent).not.toContain('undefined');
    expect(view.container.textContent).not.toContain('Missing references');
    expect(view.container.querySelector('[data-solidworks-missing-references]')).toBeNull();
  });
});
