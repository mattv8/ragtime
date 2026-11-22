using System.Diagnostics;
using System.Globalization;
using System.Threading.Tasks;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>Lifecycle of one instance. Changed only under the instance manager's lock.</summary>
    internal enum InstanceState
    {
        /// <summary>Launched or attached; the startup work has not finished.</summary>
        Starting,

        /// <summary>Session created; calls are accepted.</summary>
        Ready,

        /// <summary>Being closed (idle shutdown or host shutdown); no new calls.</summary>
        Closing,

        /// <summary>Removed from the registry; its worker is stopping.</summary>
        Stopped,
    }

    /// <summary>
    /// Everything the host knows about one instance. Fields marked "under Sync" are guarded by the manager lock
    /// (passed in as <see cref="Sync"/>). <see cref="App"/> and <see cref="Session"/> are touched only on the
    /// instance's STA thread.
    /// </summary>
    internal sealed class InstanceRecord
    {
        private volatile bool _retired;

        public InstanceRecord(int number, string mode, bool owned, int? year, Process processObject, object sync)
        {
            Number = number;
            Id = "i" + number.ToString(CultureInfo.InvariantCulture);
            Mode = mode;
            Owned = owned;
            Year = year;
            ProcessObject = processObject;
            ProcessId = processObject.Id;
            Sync = sync;
            Worker = new InstanceWorker("SolidWorks-" + Id);
        }

        /// <summary>Numeric part of the id; ids are never reused.</summary>
        public int Number { get; }

        /// <summary>Instance id such as "i1".</summary>
        public string Id { get; }

        /// <summary>"background" or "attach".</summary>
        public string Mode { get; }

        /// <summary>True when the host launched the process.</summary>
        public bool Owned { get; }

        /// <summary>
        /// Release year, or null when unknown. Owned instances know it at creation; an attached instance learns it
        /// from its readiness check. Under Sync.
        /// </summary>
        public int? Year { get; set; }

        /// <summary>Process object of the SLDWORKS process; used for exit checks and kills.</summary>
        public Process ProcessObject { get; }

        /// <summary>Process id, fixed at creation.</summary>
        public int ProcessId { get; }

        /// <summary>The manager lock.</summary>
        public object Sync { get; }

        /// <summary>STA worker thread of this instance.</summary>
        public InstanceWorker Worker { get; }

        /// <summary>Lifecycle state. Under Sync.</summary>
        public InstanceState State { get; set; } = InstanceState.Starting;

        /// <summary>True after the instance was removed from the registry. Readable from any thread.</summary>
        public bool Retired => _retired;

        /// <summary>Set once, when the instance leaves the registry.</summary>
        public void MarkRetired() => _retired = true;

        /// <summary>True while an attached call that passed its deadline is still running. Under Sync.</summary>
        public bool Stuck { get; set; }

        /// <summary>Revision number from SolidWorks, or null. Under Sync.</summary>
        public string? Revision { get; set; }

        /// <summary>Last known window visibility. Under Sync.</summary>
        public bool Visible { get; set; }

        /// <summary>Per-instance session, created on the STA thread. Read on the STA thread only.</summary>
        public ISolidWorksSession? Session { get; set; }

        /// <summary>SolidWorks application object. STA thread only.</summary>
        public SW.ISldWorks? App { get; set; }

        /// <summary>Number of times a visible titled window of this owned instance had to be hidden. STA thread only.</summary>
        public int WindowHideCount { get; set; }

        /// <summary>True after the first window check of this owned instance (the first hide). STA thread only.</summary>
        public bool FirstHideDone { get; set; }

        /// <summary>
        /// True once a titled top-level window of this owned instance was seen visible after the host's first hide:
        /// a user could have edited in it. Written on the STA thread; read by the close decision.
        /// </summary>
        public volatile bool EverVisibleAfterHide;

        /// <summary>
        /// Readiness operation of an attached instance: completes when the instance is ready (true) or failed. Null for
        /// owned instances. Under Sync (the task itself is read without the lock).
        /// </summary>
        public Task<bool>? Startup { get; set; }

        /// <summary>Monotonic milliseconds of the last call. Under Sync.</summary>
        public long LastUsedMs { get; set; }

        /// <summary>Idle seconds from the most recent request context that touched the instance. Under Sync.</summary>
        public int IdleShutdownSeconds { get; set; }
    }
}
