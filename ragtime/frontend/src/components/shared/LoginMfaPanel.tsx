import type { MfaMethod } from '@/types';
import { AuthMfaPanel } from '../AuthMfaPanel';

interface LoginMfaPanelProps {
  mode: 'verify' | 'enroll' | 'recovery';
  error: string | null;
  isLoading: boolean;
  code: string;
  rememberDevice: boolean;
  methods: MfaMethod[];
  preferredMethod: MfaMethod | null;
  mfaChallengeToken?: string;
  serverName: string;
  onCodeChange: (code: string) => void;
  onRememberDeviceChange: (remember: boolean) => void;
  onVerify: () => void;
  onSessionEstablished: () => void;
  onRecoveryContinue: () => void;
  recoveryContinueLabel?: string;
}

export function LoginMfaPanel({
  mode,
  error,
  isLoading,
  code,
  rememberDevice,
  methods,
  preferredMethod,
  mfaChallengeToken,
  serverName,
  onCodeChange,
  onRememberDeviceChange,
  onVerify,
  onSessionEstablished,
  onRecoveryContinue,
  recoveryContinueLabel,
}: LoginMfaPanelProps) {
  return (
    <AuthMfaPanel
      mode={mode}
      error={error}
      isLoading={isLoading}
      code={code}
      rememberDevice={rememberDevice}
      recoveryCodes={[]}
      {...(recoveryContinueLabel ? { recoveryContinueLabel } : {})}
      methods={methods}
      preferredMethod={preferredMethod}
      mfaChallengeToken={mfaChallengeToken}
      serverName={serverName}
      onCodeChange={onCodeChange}
      onRememberDeviceChange={onRememberDeviceChange}
      onVerify={onVerify}
      onVerified={onSessionEstablished}
      onEnrollComplete={onSessionEstablished}
      onRecoveryContinue={onRecoveryContinue}
    />
  );
}
