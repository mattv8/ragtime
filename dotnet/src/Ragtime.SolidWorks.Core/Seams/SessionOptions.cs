using System;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// What the session needs to know about how its instance is owned. Built by the instance manager (Runtime) when it
    /// creates a session, so the Readers lane never inspects the host's records itself.
    /// </summary>
    public sealed class SessionOptions
    {
        /// <summary>Creates the options.</summary>
        /// <param name="attached">True for a user's interactive SolidWorks (attach mode).</param>
        /// <param name="reapplyWindowHiding">
        /// Re-applies the host's window hiding to an owned instance. Called on the instance's STA thread. A no-op when attached.
        /// </param>
        /// <param name="log">
        /// Receives one host log line per failed script-state read or restore. Never receives script source. Optional;
        /// the default discards the lines.
        /// </param>
        public SessionOptions(bool attached, Action reapplyWindowHiding, Action<string>? log = null)
        {
            Attached = attached;
            ReapplyWindowHiding = reapplyWindowHiding ?? throw new ArgumentNullException(nameof(reapplyWindowHiding));
            Log = log ?? (_ => { });
        }

        /// <summary>True in attach mode (a user's interactive SolidWorks). Attached instances are never hidden or killed by the host.</summary>
        public bool Attached { get; }

        /// <summary>Hides the owned instance's windows again, on the instance's STA thread. A no-op when attached.</summary>
        public Action ReapplyWindowHiding { get; }

        /// <summary>Host log writer for this instance. Failed script-state reads and restores are logged here. Never throws to the caller.</summary>
        public Action<string> Log { get; }
    }
}
