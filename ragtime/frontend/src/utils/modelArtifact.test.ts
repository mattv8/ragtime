import { describe, expect, it } from 'vitest';
import {
  ArtifactRequestError,
  artifactStateFromStatus,
  formatArtifactExpiry,
  formatArtifactSize,
  parseModelArtifact,
} from './modelArtifact';

describe('artifact helpers', () => {
  it('maps HTTP status codes and retains typed errors', () => {
    expect([401, 403, 404, 410, 500].map(artifactStateFromStatus)).toEqual([
      'signin',
      'forbidden',
      'notfound',
      'expired',
      'error',
    ]);
    expect(new ArtifactRequestError(410).status).toBe(410);
  });
  it('formats sizes and singular, plural, and expired times', () => {
    const now = new Date('2026-01-01T00:00:00Z');
    expect(formatArtifactSize(1024)).toBe('1.0 KB');
    expect(formatArtifactExpiry('2026-01-01T01:00:00Z', now)).toEqual({
      label: 'Expires in 1 hour',
      expired: false,
      soon: true,
    });
    expect(formatArtifactExpiry('2026-01-03T00:00:00Z', now).label).toBe('Expires in 2 days');
    expect(formatArtifactExpiry('2025-12-31T00:00:00Z', now)).toMatchObject({
      expired: true,
      soon: false,
    });
  });
});

const id = 'a'.repeat(32);
const artifact = {
  id,
  kind: 'model',
  format: 'step',
  file_name: 'bracket.step',
  size_bytes: 123,
  expires_at: '2099-01-01T00:00:00Z',
  content_url: `/indexes/solidworks/artifacts/${id}/content`,
  page_url: `/?artifact=${id}`,
};
const output = JSON.stringify({
  ok: true,
  operation: 'model.export',
  result: { artifact, extra: 'x' },
});

describe('parseModelArtifact', () => {
  it('parses complete and truncated envelopes', () => {
    expect(parseModelArtifact({ tool: 'solidworks_host', output })).toEqual(artifact);
    expect(parseModelArtifact({ tool: 'solidworks_host', output: output.slice(0, -5) })).toEqual(
      artifact,
    );
  });
  it('rejects unsafe URLs, unrelated tools and errors', () => {
    expect(parseModelArtifact({ tool: 'other', output })).toBeNull();
    expect(
      parseModelArtifact({
        tool: 'solidworks_host',
        output: output.replace(artifact.content_url, '//evil.test'),
      }),
    ).toBeNull();
    expect(
      parseModelArtifact({
        tool: 'solidworks_host',
        output: output.replace('"ok":true', '"ok":false'),
      }),
    ).toBeNull();
  });
  it('accepts the solidworks connection type', () => {
    expect(
      parseModelArtifact({ tool: 'host', connection: { tool_type: 'solidworks' }, output }),
    ).toEqual(artifact);
  });
});
