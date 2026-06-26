import type { AuthStatus, User } from '@/types';
import { LoginCard } from './LoginPage';
import WebGLGradient from './WebGLGradient';

interface UserSpaceSharedAuthGateProps {
  authStatus: AuthStatus | null;
  onLoginSuccess: (user: User) => void;
  serverName?: string;
  detail?: string | null;
}

function fallbackAuthStatus(serverName: string): AuthStatus {
  return {
    authenticated: false,
    ldap_configured: false,
    local_admin_enabled: true,
    debug_mode: false,
    api_key_configured: false,
    session_cookie_secure: false,
    allowed_origins_open: false,
    server_name: serverName,
    authenticated_webgl_background_enabled: true,
    chat_compaction_threshold_percent: 80,
    chat_auto_compaction_threshold_percent: 99,
    auth_methods: [],
  };
}

export function UserSpaceSharedAuthGate({
  authStatus,
  onLoginSuccess,
  serverName = 'Ragtime',
  detail,
}: UserSpaceSharedAuthGateProps) {
  const resolvedAuthStatus = authStatus ?? fallbackAuthStatus(serverName);

  return (
    <div className="login-container login-gradient-container userspace-shared-auth-gate">
      <WebGLGradient className="login-background-gradient" fullscreen />
      <div className="userspace-shared-auth-card-stack">
        <div className="userspace-shared-auth-copy">
          <p className="login-subtitle">This workspace preview is protected.</p>
          <p className="login-info">
            Sign in with an allowed auth method to continue to the shared preview.
          </p>
          {detail ? <p className="login-info userspace-shared-auth-detail">{detail}</p> : null}
        </div>
        <LoginCard
          authStatus={resolvedAuthStatus}
          onLoginSuccess={onLoginSuccess}
          serverName={serverName}
        />
      </div>
    </div>
  );
}
