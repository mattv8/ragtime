import type { AppSettings } from '@/types';

type EndpointSettings = Partial<
  Pick<
    AppSettings,
    | 'openai_compatible_base_url'
    | 'llm_ollama_base_url'
    | 'llm_llama_cpp_base_url'
    | 'llm_lmstudio_base_url'
    | 'llm_omlx_base_url'
  >
>;

const LOCAL_HOSTNAMES = new Set(['localhost', 'ip6-localhost', 'ip6-loopback']);
const LOCAL_HOSTNAME_SUFFIXES = ['.localhost', '.local', '.internal', '.lan', '.home.arpa'];

function parseIpv4(host: string): number[] | null {
  const parts = host.split('.');
  if (parts.length !== 4) return null;
  const octets = parts.map((part) => (/^\d{1,3}$/.test(part) ? Number(part) : NaN));
  return octets.every((octet) => Number.isInteger(octet) && octet <= 255) ? octets : null;
}

function isNonPublicIpv4([a, b]: number[]): boolean {
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    a >= 224
  );
}

function isNonPublicIpv6(host: string): boolean {
  if (host === '::' || host === '::1') return true;
  const mappedIpv4 = host.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/);
  if (mappedIpv4) {
    const octets = parseIpv4(mappedIpv4[1]);
    return octets ? isNonPublicIpv4(octets) : false;
  }
  const mappedHex = host.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (mappedHex) {
    const high = parseInt(mappedHex[1], 16);
    const low = parseInt(mappedHex[2], 16);
    return isNonPublicIpv4([high >> 8, high & 0xff, low >> 8, low & 0xff]);
  }
  const firstHextet = parseInt(host.split(':')[0] || '0', 16);
  return (
    (firstHextet & 0xfe00) === 0xfc00 || // unique local fc00::/7
    (firstHextet & 0xffc0) === 0xfe80 || // link-local fe80::/10
    (firstHextet & 0xff00) === 0xff00 // multicast ff00::/8
  );
}

/**
 * Returns true when a model endpoint URL targets a host that appears to be
 * reachable on the public internet (not loopback, private, or LAN-only).
 */
export function isPublicModelEndpoint(baseUrl: string | null | undefined): boolean {
  const raw = (baseUrl ?? '').trim();
  if (!raw) return false;

  let hostname: string;
  try {
    hostname = new URL(raw).hostname;
  } catch {
    // Unparseable endpoints cannot be classified; err toward disclosing the risk.
    return true;
  }

  const host = hostname
    .toLowerCase()
    .replace(/^\[|\]$/g, '')
    .replace(/\.$/, '');
  if (!host) return false;
  if (LOCAL_HOSTNAMES.has(host)) return false;
  if (LOCAL_HOSTNAME_SUFFIXES.some((suffix) => host.endsWith(suffix))) return false;

  if (host.includes(':')) return !isNonPublicIpv6(host);

  const ipv4 = parseIpv4(host);
  if (ipv4) return !isNonPublicIpv4(ipv4);

  // Single-label names (Docker service names, LAN hosts) are not public.
  return host.includes('.');
}

/**
 * Untrusted model endpoints: any configured generic OpenAI-compatible
 * provider, or a public Ollama, llama.cpp, LM Studio, or oMLX chat endpoint.
 */
export function hasUntrustedModelEndpoint(settings: EndpointSettings): boolean {
  if (settings.openai_compatible_base_url?.trim()) return true;
  return [
    settings.llm_ollama_base_url,
    settings.llm_llama_cpp_base_url,
    settings.llm_lmstudio_base_url,
    settings.llm_omlx_base_url,
  ].some(isPublicModelEndpoint);
}
