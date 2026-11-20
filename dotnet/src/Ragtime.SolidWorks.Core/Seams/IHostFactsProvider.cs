namespace Ragtime.SolidWorks.Core
{
    /// <summary>Facts about the host process and its account, for <c>status</c>. Implemented by the Host lane.</summary>
    public interface IHostFactsProvider
    {
        /// <summary>Returns the current facts. Called on each <c>status</c> request; must not block for long.</summary>
        HostFacts GetHostFacts();
    }

    /// <summary>Snapshot of host facts.</summary>
    public sealed class HostFacts
    {
        /// <summary>Creates the facts.</summary>
        public HostFacts(string accountName, string sid, bool isAdministrator, int sessionId, bool interactive, double uptimeSeconds)
        {
            AccountName = accountName;
            Sid = sid;
            IsAdministrator = isAdministrator;
            SessionId = sessionId;
            Interactive = interactive;
            UptimeSeconds = uptimeSeconds;
        }

        /// <summary>DOMAIN\user of the process account.</summary>
        public string AccountName { get; }

        /// <summary>SID of the process account.</summary>
        public string Sid { get; }

        /// <summary>True when the process token has the Administrators role.</summary>
        public bool IsAdministrator { get; }

        /// <summary>Windows session id of the process.</summary>
        public int SessionId { get; }

        /// <summary>True when the process runs in an interactive desktop session.</summary>
        public bool Interactive { get; }

        /// <summary>Seconds since the process started.</summary>
        public double UptimeSeconds { get; }
    }
}
