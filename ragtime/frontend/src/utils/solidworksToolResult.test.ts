import { describe, expect, it } from 'vitest';
import {
  humanizeArguments,
  humanizeOperation,
  parseSolidWorksResult,
  solidWorksToolPillName,
} from './solidworksToolResult';

describe('SolidWorks results', () => {
  it('brand-cases fallback pill names and preserves configured names', () => {
    expect(solidWorksToolPillName({ tool: 'solidworks_solidworks_render_pc_dev' })).toBe(
      'SolidWorks render pc dev',
    );
    expect(
      solidWorksToolPillName({
        tool: 'solidworks_test',
        connection: { tool_config_name: 'Custom SW' },
      }),
    ).toBe('Custom SW');
  });
  const events = [
    '{"protocol":1,"id":"6db68cda-c820-4b24-904e-42eb3f2b3315","operation":"guide","ok":true,"result":{"topic":"overview"},"_ragtime_output_truncated":true}',
    '{"protocol":1,"id":"2677bdd8-1187-41ba-a3cb-8466bba63a1a","operation":"guide","ok":true,"result":{"topic":"previews"},"_ragtime_output_truncated":true}',
    '{"protocol":1,"id":"dbc322ee-bdf9-4bb4-b67f-97853841218f","ok":true,"operation":"describe","result":{"protocol":1,"host_version":"0.5.0","operations":[{"name":"describe","summary":"Lists every operation',
    '{"protocol":1,"id":"c2b83c2e-874b-41cc-b78b-849861082fa3","ok":true,"operation":"documents.list","result":{"items":[{"document_id":"doc-1","title":"box_with_lid"}],"total":4},"elapsed_ms":12}',
    '{"protocol":1,"id":"a66847ca-7baa-4139-9f94-a485cfcf8c93","operation":"model.preview","ok":false,"error":{"code":"invalid_arguments","message":"\'path\' is not an argument of this operation.","details":{"field":"path"}}}',
    '{"protocol":1,"ok":true,"operation":"model.preview","result":{"artifact":{"kind":"preview","format":"png","file_name":"box_with_lid.png"}}}',
    '{"protocol":1,"ok":true,"operation":"model.export","result":{"artifact":{"kind":"model","format":"step","file_name":"box_with_lid.step"}}}',
  ];
  it.each([
    [0, 'guide', 'ok'],
    [1, 'guide', 'ok'],
    [2, 'describe', 'ok'],
    [3, 'documents.list', 'ok'],
    [4, 'model.preview', 'error'],
    [5, 'model.preview', 'ok'],
    [6, 'model.export', 'ok'],
  ])('parses live event %i', (index, operation, state) => {
    expect(parseSolidWorksResult(events[index as number], { operation })).toMatchObject({
      operation,
      state,
    });
  });

  it('handles marker, cut, regex, unparsed and running output', () => {
    expect(
      parseSolidWorksResult(
        '{"operation":"guide","ok":true,"_ragtime_output_truncated":true}',
        null,
      ).truncated,
    ).toBe('marker');
    const cut = parseSolidWorksResult(
      '{"operation":"describe","ok":true,"result":{"items":[{"name":"one"},{"name":"par',
      null,
    );
    expect(cut).toMatchObject({
      operation: 'describe',
      state: 'ok',
      truncated: 'cut',
      result: { items: [{ name: 'one' }] },
    });
    expect(
      parseSolidWorksResult(
        'prefix "operation":"model.export", "ok":false, "error":{"code":"bad","message":"Oops"}',
        null,
      ),
    ).toMatchObject({
      state: 'error',
      operation: 'model.export',
      result: null,
      error: { code: 'bad', message: 'Oops' },
    });
    expect(parseSolidWorksResult('not json', null).state).toBe('unparsed');
    expect(parseSolidWorksResult('', null, 'complete').state).toBe('unparsed');
    expect(parseSolidWorksResult('{"ok":false}', null)).toMatchObject({
      state: 'error',
      error: null,
    });
    expect(parseSolidWorksResult(null, { operation: 'status' }, 'running')).toMatchObject({
      state: 'running',
      operation: 'status',
    });
  });

  it('extracts metadata, timeout and error details', () => {
    const output = JSON.stringify({
      ok: false,
      error: { code: 'failed', message: 'Oops', details: { hresult: '0x123', field: 'format' } },
      host: { host_version: '0.5', solidworks_version: 2024 },
      elapsed_ms: 12,
    });
    expect(
      parseSolidWorksResult(output, {
        operation: 'model.export',
        arguments: { document: 'doc-1' },
        timeout: 300,
      }),
    ).toMatchObject({
      operation: 'model.export',
      elapsedMs: 12,
      solidworksVersion: '2024',
      args: [
        { key: 'document', label: 'Document', value: 'doc-1', kind: 'code' },
        { key: 'timeout', label: 'Timeout', value: '300 s', kind: 'number' },
      ],
      error: {
        details: [
          { label: 'HRESULT', value: '0x123' },
          { label: 'Field', value: 'format' },
        ],
      },
    });
  });

  it.each([
    ['describe', 'Describe operations'],
    ['status', 'Status'],
    ['documents.list', 'List documents'],
    ['document.open', 'Open document'],
    ['document.close', 'Close document'],
    ['document.copy', 'Copy model'],
    ['model.export', 'Export model'],
    ['model.preview', 'Preview model'],
    ['model.summary', 'Model summary'],
    ['components.list', 'List components'],
    ['features.list', 'List features'],
    ['configurations.list', 'List configurations'],
    ['dimensions.list', 'List dimensions'],
    ['mates.list', 'List mates'],
    ['equations.list', 'List equations'],
    ['properties.get', 'Read properties'],
    ['mass_properties.get', 'Mass properties'],
    ['feature.get', 'Read feature'],
    ['sketch.get', 'Read sketch'],
    ['script.run', 'Run script'],
    ['pdm.search', 'Search PDM'],
    ['pdm.lookup', 'Look up PDM documents'],
    ['pdm.file.get_latest', 'PDM get latest'],
    ['other.some_thing', 'Other some thing'],
    ['', 'SolidWorks'],
  ])('humanizes %s', (operation, title) => expect(humanizeOperation(operation)).toBe(title));

  it('humanizes guide and arguments', () => {
    expect(humanizeOperation('guide')).toBe('List guide topics');
    expect(humanizeOperation('guide', { topic: 'previews' })).toBe('Read guide: previews');
    expect(
      humanizeArguments({
        include_preview: true,
        enabled: false,
        width: 1200,
        path: 'C:\\part.sldprt',
        posix: '/tmp/a',
        document: 'doc-1',
        view: 'isometric',
        format: 'step',
        nested: { x: 1 },
        empty: null,
      }),
    ).toEqual([
      { key: 'include_preview', label: 'Include preview', value: 'Yes', kind: 'boolean' },
      { key: 'enabled', label: 'Enabled', value: 'No', kind: 'boolean' },
      { key: 'width', label: 'Width', value: '1200', kind: 'number' },
      { key: 'path', label: 'Path', value: 'C:\\part.sldprt', kind: 'code' },
      { key: 'posix', label: 'Posix', value: '/tmp/a', kind: 'code' },
      { key: 'document', label: 'Document', value: 'doc-1', kind: 'code' },
      { key: 'view', label: 'View', value: 'Isometric', kind: 'text' },
      { key: 'format', label: 'Format', value: 'STEP', kind: 'text' },
      { key: 'nested', label: 'Nested', value: '{"x":1}', kind: 'code' },
    ]);
  });
});
