using System;
using System.Collections.Generic;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>
    /// The logged-in vault sessions of this process, one per vault name. <see cref="Get"/> and <see cref="ReleaseAll"/> run
    /// only on the PDM thread; <see cref="LoggedIn"/> may be read from any thread and never waits for a login.
    /// </summary>
    internal sealed class PdmSessions
    {
        private readonly Dictionary<string, PdmVault> _vaults = new Dictionary<string, PdmVault>(StringComparer.OrdinalIgnoreCase);
        private readonly Action<string> _log;
        private readonly object _gate = new object();
        private IReadOnlyDictionary<string, string?> _loggedIn = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

        /// <summary>Creates the cache; <paramref name="log"/> receives the write-call log lines.</summary>
        public PdmSessions(Action<string> log)
        {
            _log = log;
        }

        /// <summary>Returns the session of a vault, logging in on first use. A session PDM no longer reports is replaced.</summary>
        public PdmVault Get(string vault)
        {
            if (_vaults.TryGetValue(vault, out var session))
            {
                if (session.IsLoggedIn)
                {
                    return session;
                }

                _vaults.Remove(vault);
                session.Release();
                Publish();
            }

            session = PdmVaultLogin.Open(vault, _log);
            _vaults[vault] = session;
            Publish();
            return session;
        }

        /// <summary>User name per vault this process holds a logged-in session for. The returned map is never modified.</summary>
        public IReadOnlyDictionary<string, string?> LoggedIn()
        {
            lock (_gate)
            {
                return _loggedIn;
            }
        }

        /// <summary>Releases every session. Runs on the PDM thread when it ends.</summary>
        public void ReleaseAll()
        {
            foreach (var session in _vaults.Values)
            {
                session.Release();
            }

            _vaults.Clear();
            Publish();
        }

        private void Publish()
        {
            var snapshot = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
            foreach (var session in _vaults.Values)
            {
                snapshot[session.Vault] = session.User;
            }

            lock (_gate)
            {
                _loggedIn = snapshot;
            }
        }
    }
}
