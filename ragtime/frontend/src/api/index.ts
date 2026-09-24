export {
  api,
  ApiError,
  apiFetch,
  beginResponseSessionEstablishment,
  getResponseAuthContext,
  isResponseAuthContextCurrent,
} from './client';
export type {
  AuthPurpose,
  RequestAuthContext,
  SessionLifecycleEvent,
  SessionPhase,
} from '@/auth/sessionLifecycle';
export type {
  ChatTaskStreamEvent,
  DockerContainer,
  DockerNetwork,
  DockerDiscoveryResponse,
  DockerSSHConfig,
} from './client';
