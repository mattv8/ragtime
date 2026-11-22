using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Where a unit of work should run. Created with the static factories.</summary>
    public enum InstanceTargetKind
    {
        /// <summary>A specific instance (background or attached), identified by its instance id.</summary>
        Specific,

        /// <summary>Launch a new background instance (the pool decided that one is needed).</summary>
        StartNew,

        /// <summary>Attach mode: select (or attach to) the SolidWorks instance in the host's session.</summary>
        Attach,
    }

    /// <summary>Target of <see cref="ISolidWorksInstanceManager.RunAsync{T}"/>.</summary>
    public sealed class InstanceTarget
    {
        private InstanceTarget(InstanceTargetKind kind, string? instanceId, string? retireInstanceId = null)
        {
            Kind = kind;
            InstanceId = instanceId;
            RetireInstanceId = retireInstanceId;
        }

        /// <summary>Kind of target.</summary>
        public InstanceTargetKind Kind { get; }

        /// <summary>Instance id for <see cref="InstanceTargetKind.Specific"/>; otherwise null.</summary>
        public string? InstanceId { get; }

        /// <summary>
        /// For <see cref="InstanceTargetKind.StartNew"/> only: an owned idle instance of another pinned year, chosen by
        /// Core (<see cref="InstanceRoom.InstanceToRetire"/>), to close first so the launch stays within max_instances.
        /// Null when no instance has to make room.
        /// </summary>
        public string? RetireInstanceId { get; }

        /// <summary>Run on the named instance.</summary>
        public static InstanceTarget Specific(string instanceId)
        {
            if (string.IsNullOrEmpty(instanceId))
            {
                throw new ArgumentException("An instance id is required.", nameof(instanceId));
            }

            return new InstanceTarget(InstanceTargetKind.Specific, instanceId);
        }

        /// <summary>Start a new background instance and run there.</summary>
        public static InstanceTarget StartNew() => new InstanceTarget(InstanceTargetKind.StartNew, null);

        /// <summary>
        /// Start a new background instance after closing <paramref name="retireInstanceId"/> (an idle owned instance of
        /// another pinned year, chosen by Core). The close counts against the request deadline.
        /// </summary>
        public static InstanceTarget StartNew(string? retireInstanceId) =>
            new InstanceTarget(InstanceTargetKind.StartNew, null, retireInstanceId);

        /// <summary>Attach mode target.</summary>
        public static InstanceTarget Attach() => new InstanceTarget(InstanceTargetKind.Attach, null);
    }

    /// <summary>Point-in-time facts about one instance, as the instance manager reports them.</summary>
    public sealed class InstanceSnapshot
    {
        /// <summary>Creates a snapshot.</summary>
        public InstanceSnapshot(
            string instanceId,
            string mode,
            bool owned,
            int? version,
            string? revision,
            int processId,
            bool visible,
            bool busy,
            int documentCount)
        {
            InstanceId = instanceId;
            Mode = mode;
            Owned = owned;
            Version = version;
            Revision = revision;
            ProcessId = processId;
            Visible = visible;
            Busy = busy;
            DocumentCount = documentCount;
        }

        /// <summary>Host-assigned id such as "i1".</summary>
        public string InstanceId { get; }

        /// <summary>"background" or "attach".</summary>
        public string Mode { get; }

        /// <summary>True when the host launched the process (only those are ever terminated by the host).</summary>
        public bool Owned { get; }

        /// <summary>Release year, or null when unknown.</summary>
        public int? Version { get; }

        /// <summary>Revision number such as "32.5.0", or null when unknown.</summary>
        public string? Revision { get; }

        /// <summary>Process id of the SLDWORKS process.</summary>
        public int ProcessId { get; }

        /// <summary>Window visibility.</summary>
        public bool Visible { get; }

        /// <summary>True while a call runs on the instance or a timed-out call has not returned.</summary>
        public bool Busy { get; }

        /// <summary>Documents the host tracks in this instance. The instance manager may leave it 0; Core overwrites it.</summary>
        public int DocumentCount { get; }

        /// <summary>Copy with <see cref="DocumentCount"/> replaced.</summary>
        public InstanceSnapshot WithDocumentCount(int documentCount) =>
            new InstanceSnapshot(InstanceId, Mode, Owned, Version, Revision, ProcessId, Visible, Busy, documentCount);
    }

    /// <summary>Result of <see cref="ISolidWorksInstanceManager.RunAsync{T}"/>: the value and the instance it ran on.</summary>
    public sealed class InstanceRun<T>
    {
        /// <summary>Creates the run result.</summary>
        public InstanceRun(InstanceSnapshot instance, T value)
        {
            Instance = instance;
            Value = value;
        }

        /// <summary>Snapshot of the instance after the work finished (feeds the envelope's host block).</summary>
        public InstanceSnapshot Instance { get; }

        /// <summary>Value returned by the work delegate.</summary>
        public T Value { get; }
    }

    /// <summary>
    /// Owns the SolidWorks instances and their STA threads. Implemented by the Runtime lane
    /// (Ragtime.SolidWorks.Runtime) and injected into <see cref="RequestProcessor"/>.
    ///
    /// Errors (all as <see cref="HostErrorException"/>):
    /// version_not_installed (pinned year missing, background mode), version_mismatch and solidworks_not_running
    /// (attach mode; see <see cref="AttachSelection"/>), solidworks_start_failed, instance_limit (only
    /// via <see cref="InstancePool"/>), instance_timeout (instance not ready in time, or an attached instance is
    /// still stuck), operation_timeout (deadline passed; details.instance_restarted tells whether a background
    /// instance was killed). solidworks_not_running is also thrown for a specific instance id that no longer exists.
    /// Exceptions thrown by the work delegate pass through unchanged.
    ///
    /// Threading: callable from any thread. Work runs on the instance's STA thread; calls to one instance are
    /// serialized, calls to different instances may run in parallel.
    /// </summary>
    public interface ISolidWorksInstanceManager
    {
        /// <summary>
        /// After a timed-out script: probes the instance with a short call on its own work item, limited to 10 s.
        /// Returns ScriptRecovery.ProbeOk when it answered. When it did not answer:
        /// - an owned instance: kill its SLDWORKS process (pid and start time verified), retire the record, raise InstanceStopped,
        ///   and return ScriptRecovery.Restarted;
        /// - an attached instance: mark it stuck, never kill it, and return ScriptRecovery.Stuck.
        /// </summary>
        Task<string> RecoverAfterScriptAsync(string instanceId, CancellationToken cancellationToken);

        /// <summary>Installed SolidWorks release years, ascending. Empty when none are installed. Never starts anything.</summary>
        IReadOnlyList<int> GetInstalledVersions();

        /// <summary>Current instances (background and attached). Never starts or attaches anything.</summary>
        IReadOnlyList<InstanceSnapshot> GetInstances();

        /// <summary>
        /// Runs <paramref name="work"/> on the target instance's STA thread. <paramref name="deadline"/> is the
        /// single deadline of the whole request, created by the caller when the request started: startup, queue
        /// wait and the call all share its remaining time, and an implementation must not start its own.
        /// <paramref name="context"/> supplies the version pin, instance mode, max_instances and
        /// idle_shutdown_seconds. Returns the value and a snapshot of the instance.
        /// </summary>
        Task<InstanceRun<T>> RunAsync<T>(
            InstanceTarget target,
            RequestContext context,
            Deadline deadline,
            Func<ISolidWorksSession, T> work,
            CancellationToken cancellationToken);

        /// <summary>
        /// Raised with the instance id when an instance is gone (exited, killed, or closed by idle shutdown).
        /// Core forgets its documents. Raised on any thread.
        /// </summary>
        event Action<string>? InstanceStopped;

        /// <summary>Closes every background instance the host owns (and releases its licence). Attached instances are never touched.</summary>
        Task ShutdownAsync(CancellationToken cancellationToken);
    }
}
