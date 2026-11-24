import { describe, expect, it } from 'vitest';
import { getModelArtifactRouteParam } from './App';

const id = 'a1b2c3d4'.repeat(4);

describe('getModelArtifactRouteParam', () => {
  it('returns the artifact query value, valid or not', () => {
    expect(getModelArtifactRouteParam(`?artifact=${id}`)).toBe(id);
    expect(getModelArtifactRouteParam(`?view=chat&artifact=${id}`)).toBe(id);
    expect(getModelArtifactRouteParam('?artifact=abc')).toBe('abc');
    expect(getModelArtifactRouteParam('?artifact=')).toBe('');
  });

  it('returns null when the artifact page is not requested', () => {
    expect(getModelArtifactRouteParam('')).toBeNull();
    expect(getModelArtifactRouteParam('?view=chat')).toBeNull();
  });
});
