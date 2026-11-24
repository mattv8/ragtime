import { describe, expect, it } from 'vitest';
import type { HeartbeatStatus, ToolConfig } from '@/types';
import { summarizeToolGroupHealth } from './toolGroupHealth';

const enabled = (id: string) => ({ id, enabled: true }) as ToolConfig;
const disabled = (id: string) => ({ id, enabled: false }) as ToolConfig;
const heartbeat = (alive: boolean) => ({ alive }) as HeartbeatStatus;

describe('summarizeToolGroupHealth', () => {
  it('handles empty and all disabled groups', () => {
    expect(summarizeToolGroupHealth([], {}, null)).toEqual({ tone: 'neutral', label: 'Empty' });
    expect(summarizeToolGroupHealth([disabled('a')], { a: heartbeat(false) }, null)).toEqual({
      tone: 'neutral',
      label: 'All disabled',
    });
  });
  it('prioritizes failures over checking and ignores disabled and testing failures', () => {
    expect(
      summarizeToolGroupHealth(
        [enabled('a'), enabled('b'), enabled('c'), disabled('d')],
        { a: heartbeat(false), b: heartbeat(false), d: heartbeat(false) },
        'b',
      ),
    ).toEqual({ tone: 'error', label: '1 failing' });
    expect(summarizeToolGroupHealth([enabled('a')], { a: heartbeat(false) }, 'a')).toEqual({
      tone: 'neutral',
      label: 'Checking…',
    });
  });
  it('reports missing heartbeats and healthy groups', () => {
    expect(summarizeToolGroupHealth([enabled('a')], {}, null)).toEqual({
      tone: 'neutral',
      label: 'Checking…',
    });
    expect(
      summarizeToolGroupHealth(
        [enabled('a'), disabled('b')],
        { a: heartbeat(true), b: heartbeat(false) },
        null,
      ),
    ).toEqual({ tone: 'success', label: 'All healthy' });
  });
});
