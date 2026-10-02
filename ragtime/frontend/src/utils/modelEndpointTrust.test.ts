import { describe, expect, it } from 'vitest';
import { hasUntrustedModelEndpoint, isPublicModelEndpoint } from './modelEndpointTrust';

const localDefaults = {
  llm_ollama_base_url: 'http://localhost:11434',
  llm_llama_cpp_base_url: 'http://host.docker.internal:8080',
  llm_lmstudio_base_url: 'http://host.docker.internal:1234',
  llm_omlx_base_url: 'http://host.docker.internal:8000',
};

describe('isPublicModelEndpoint', () => {
  it.each([
    '',
    'http://localhost:11434',
    'http://127.0.0.1:11434',
    'http://host.docker.internal:1234',
    'http://gateway.docker.internal:1234',
    'http://ollama:11434',
    'http://studio.local:1234',
    'http://box.lan:1234',
    'http://10.0.0.5:8000',
    'http://172.20.1.2:8000',
    'http://192.168.1.20:11434',
    'http://169.254.1.1:11434',
    'http://100.100.1.1:11434',
    'http://0.0.0.0:11434',
    'http://[::1]:11434',
    'http://[fd12:3456::1]:11434',
    'http://[fe80::1]:11434',
    'http://[::ffff:192.168.1.5]:11434',
  ])('treats %s as non-public', (url) => {
    expect(isPublicModelEndpoint(url)).toBe(false);
  });

  it.each([
    'https://ollama.example.com',
    'http://8.8.8.8:11434',
    'http://172.32.0.1:8000',
    'http://[2001:4860::1]:11434',
    'http://[::ffff:8.8.8.8]:11434',
    'http://[64:ff9b::8.8.8.8]:11434',
    'not a url',
  ])('treats %s as public', (url) => {
    expect(isPublicModelEndpoint(url)).toBe(true);
  });
});

describe('hasUntrustedModelEndpoint', () => {
  it('is false for trusted providers with local self-hosted defaults', () => {
    expect(hasUntrustedModelEndpoint({ ...localDefaults, openai_compatible_base_url: '' })).toBe(
      false,
    );
  });

  it('is false when no endpoints are present', () => {
    expect(hasUntrustedModelEndpoint({})).toBe(false);
  });

  it.each(['http://192.168.1.10:8080/v1', 'http://localhost:8080/v1'])(
    'is true whenever a generic OpenAI-compatible provider is configured (%s)',
    (url) => {
      expect(hasUntrustedModelEndpoint({ ...localDefaults, openai_compatible_base_url: url })).toBe(
        true,
      );
    },
  );

  it.each([
    'llm_ollama_base_url',
    'llm_llama_cpp_base_url',
    'llm_lmstudio_base_url',
    'llm_omlx_base_url',
  ] as const)('is true when %s points to a public host', (field) => {
    expect(
      hasUntrustedModelEndpoint({ ...localDefaults, [field]: 'https://models.example.com' }),
    ).toBe(true);
  });
});
