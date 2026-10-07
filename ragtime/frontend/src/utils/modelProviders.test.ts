import { describe, expect, it } from 'vitest';

import { resolveProviderModelSelection } from './modelProviders';

describe('resolveProviderModelSelection', () => {
  it('does not resolve an explicit compatible-provider model to another provider', () => {
    const selection = resolveProviderModelSelection('openai_compatible::shared-id', [
      { id: 'shared-id', provider: 'openai' },
    ]);

    expect(selection.matchedModel).toBeUndefined();
  });

  it('does not resolve an explicit non-compatible model to the compatible provider', () => {
    const selection = resolveProviderModelSelection('openai::shared-id', [
      { id: 'shared-id', provider: 'openai_compatible' },
    ]);

    expect(selection.matchedModel).toBeUndefined();
  });

  it('does not resolve an explicit openai_compatible org/model to generic model id', () => {
    const selection = resolveProviderModelSelection('openai_compatible::org/model', [
      { id: 'model', provider: 'openai' },
    ]);

    expect(selection.matchedModel).toBeUndefined();
  });
});
