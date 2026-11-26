using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>
    /// Owns the SolidWorks instances: launches background instances the host owns, attaches to running instances in
    /// the host's session, runs work on each instance's STA thread within the request's deadline, and shuts instances down.
    /// Instance states: Starting -> Ready -> Closing -> Stopped (see the runtime report). Every transition to Stopped
    /// goes through <see cref="Retire"/>, so <see cref="InstanceStopped"/> is raised exactly once per instance.
    /// Attached instances: one readiness operation per process (<see cref="InstanceRecord.Startup"/>) lives until it
    /// finishes; callers wait for it at most min(3 s, remaining budget).
    /// </summary>
    public sealed class SolidWorksInstanceManager : ISolidWorksInstanceManager, IDisposable
    {
        private const int TickMilliseconds = 1000;
        private const long IdleCheckMilliseconds = 15000;
        private const int AttachReadyMilliseconds = 3000;
        private const int SaveCheckMilliseconds = 5000;
        private const int CloseGraceMilliseconds = 20000;
        private const int ShutdownWaitMilliseconds = 60000;
        private const int ShutdownPollMilliseconds = 200;
        private const int ScriptProbeMilliseconds = 10000;

        private readonly ISolidWorksSessionFactory _sessionFactory;
        private readonly Action<string> _log;
        private readonly IMonotonicClock _clock = new SystemMonotonicClock();
        private readonly object _lock = new object();
        private readonly Dictionary<string, InstanceRecord> _records = new Dictionary<string, InstanceRecord>();
        private readonly Timer _timer;
        private int _nextNumber;
        private int _launching;
        private int _ticking;
        private long _lastIdleCheckMs;
        private bool _shuttingDown;
        private volatile bool _disposed;

        /// <summary>Creates the manager. <paramref name="log"/> receives lifecycle lines; it may be called from any thread.</summary>
        public SolidWorksInstanceManager(ISolidWorksSessionFactory sessionFactory, Action<string> log)
        {
            _sessionFactory = sessionFactory ?? throw new ArgumentNullException(nameof(sessionFactory));
            _log = log ?? throw new ArgumentNullException(nameof(log));
            _timer = new Timer(OnTick, null, TickMilliseconds, TickMilliseconds);
        }

        /// <inheritdoc />
        public event Action<string>? InstanceStopped;

        /// <inheritdoc />
        public IReadOnlyList<int> GetInstalledVersions() =>
            SolidWorksInstallations.Discover().Select(install => install.Year).OrderBy(year => year).ToList();

        /// <inheritdoc />
        public IReadOnlyList<InstanceSnapshot> GetInstances()
        {
            lock (_lock)
            {
                return _records.Values
                    .Where(IsLive)
                    .OrderBy(record => record.Number)
                    .Select(SnapshotLocked)
                    .ToList();
            }
        }

        /// <summary>
        /// Paths of the documents open in the instances the host started, for the copy retention sweep. Each such instance
        /// answers on its own STA thread within <paramref name="timeout"/>. Returns null when one of them does not answer or
        /// cannot be listed (the caller then keeps every copy). Attached instances are not asked. Never starts or attaches.
        /// </summary>
        public IReadOnlyList<string?>? OpenDocumentPathsOfOwnedInstances(TimeSpan timeout)
        {
            var items = new List<WorkItem<IReadOnlyList<string?>?>>();
            lock (_lock)
            {
                foreach (var record in _records.Values.Where(candidate => candidate.Owned && candidate.State != InstanceState.Stopped))
                {
                    var item = new WorkItem<IReadOnlyList<string?>?>(() => InstanceStartup.OpenDocumentPaths(record, _log), null);
                    if (record.Stuck || !record.Worker.TryEnqueue(item))
                    {
                        AbandonAll(items);
                        return null;
                    }

                    items.Add(item);
                }
            }

            try
            {
                var tasks = items.Select(item => item.Task).ToArray();
                if (!Task.WaitAll(tasks, timeout))
                {
                    AbandonAll(items);
                    return null;
                }

                var paths = new List<string?>();
                foreach (var task in tasks)
                {
                    var listed = task.Result;
                    if (listed == null)
                    {
                        return null;
                    }

                    paths.AddRange(listed);
                }

                return paths;
            }
            catch (AggregateException)
            {
                return null;
            }
        }

        private static void AbandonAll(IEnumerable<WorkItem<IReadOnlyList<string?>?>> items)
        {
            foreach (var item in items)
            {
                item.TryAbandon();
            }
        }

        /// <inheritdoc />
        public async Task<InstanceRun<T>> RunAsync<T>(
            InstanceTarget target,
            RequestContext context,
            Deadline deadline,
            Func<ISolidWorksSession, T> work,
            CancellationToken cancellationToken)
        {
            if (target == null)
            {
                throw new ArgumentNullException(nameof(target));
            }

            if (context == null)
            {
                throw new ArgumentNullException(nameof(context));
            }

            if (deadline == null)
            {
                throw new ArgumentNullException(nameof(deadline));
            }

            if (work == null)
            {
                throw new ArgumentNullException(nameof(work));
            }

            cancellationToken.ThrowIfCancellationRequested();

            var record = await ResolveAsync(target, context, deadline, cancellationToken).ConfigureAwait(false);

            var call = new CallMarker();
            var item = new WorkItem<T>(() => RunOnWorker(record, work), () => ClearStuck(record, call));
            EnqueueOnRecord(record, item, context);

            var outcome = await DeadlineWait.WaitAsync(item.Task, deadline, cancellationToken).ConfigureAwait(false);
            switch (outcome)
            {
                case WaitOutcome.Completed:
                {
                    var value = await item.Task.ConfigureAwait(false);
                    return new InstanceRun<T>(CompleteRun(record), value);
                }

                case WaitOutcome.Cancelled:
                    // A queued call is dropped; a running call finishes on its own and its result is discarded.
                    item.TryAbandon();
                    throw new OperationCanceledException(cancellationToken);

                case WaitOutcome.TimedOut:
                    throw await TimeoutErrorAsync(record, item, call, deadline).ConfigureAwait(false);

                default:
                    throw new InvalidOperationException("Unknown wait outcome.");
            }
        }

        /// <inheritdoc />
        /// <remarks>
        /// The probe is a call on the instance's own worker, limited to <see cref="ScriptProbeMilliseconds"/>. An owned
        /// instance that does not answer is killed through the same path as a call timeout, but only after its pid and start
        /// time match the record. An attached instance is marked stuck and left alone.
        /// </remarks>
        public async Task<string> RecoverAfterScriptAsync(string instanceId, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            InstanceRecord? record;
            lock (_lock)
            {
                record = _records.TryGetValue(instanceId ?? string.Empty, out var found) && IsLive(found) ? found : null;
            }

            if (record == null)
            {
                Log($"{instanceId}: after a script timeout the instance is already gone");
                return ScriptRecovery.Restarted;
            }

            var call = new CallMarker();
            var probe = new WorkItem<bool>(() => ProbeOnWorker(record), () => ClearStuck(record, call));
            var enqueued = record.Worker.TryEnqueue(probe);
            var outcome = enqueued
                ? await DeadlineWait.WaitAsync(probe.Task, TimeSpan.FromMilliseconds(ScriptProbeMilliseconds), cancellationToken).ConfigureAwait(false)
                : WaitOutcome.TimedOut;
            if (outcome == WaitOutcome.Cancelled)
            {
                probe.TryAbandon();
                throw new OperationCanceledException(cancellationToken);
            }

            if (outcome == WaitOutcome.Completed && probe.Task.Status == TaskStatus.RanToCompletion)
            {
                Log($"{record.Id}: answered the probe after a script");
                return ScriptRecovery.ProbeOk;
            }

            // A queued probe is skipped; a running one keeps the instance busy until it returns.
            probe.TryAbandon();
            if (record.Owned)
            {
                Log($"{record.Id}: did not answer the probe after a script; stopping pid {record.ProcessId}");
                if (SameProcessAsRecord(record))
                {
                    await KillAndFinishAsync(record, "no answer after a script").ConfigureAwait(false);
                }
                else
                {
                    Log($"{record.Id}: pid {record.ProcessId} is not the process the host started; it was not killed");
                    Finish(record, "no answer after a script; process not started by the SolidWorks agent");
                }

                return ScriptRecovery.Restarted;
            }

            lock (_lock)
            {
                // Same rule as a call timeout: busy while the call that passed its deadline is still running.
                if (!call.Finished)
                {
                    record.Stuck = true;
                }
            }

            Log($"{record.Id}: did not answer the probe after a script on an attached instance; marked busy, not stopped");
            return ScriptRecovery.Stuck;
        }

        /// <inheritdoc />
        public async Task ShutdownAsync(CancellationToken cancellationToken)
        {
            var toClose = new List<InstanceRecord>();
            var attached = new List<InstanceRecord>();
            var owned = new List<InstanceRecord>();
            lock (_lock)
            {
                _shuttingDown = true;
                _timer.Change(Timeout.Infinite, Timeout.Infinite);
                foreach (var record in _records.Values)
                {
                    if (!record.Owned)
                    {
                        attached.Add(record);
                        continue;
                    }

                    owned.Add(record);
                    if (record.State == InstanceState.Starting || record.State == InstanceState.Ready)
                    {
                        record.State = InstanceState.Closing;
                        toClose.Add(record);
                    }
                }
            }

            Log($"shutdown: closing {toClose.Count} owned instance(s), releasing {attached.Count} attached instance(s)");
            foreach (var record in toClose)
            {
                RunDetached(() => CloseAndFinishAsync(record, "SolidWorks agent shutdown"));
            }

            foreach (var record in attached)
            {
                Finish(record, "SolidWorks agent shutdown");
            }

            var limit = DateTime.UtcNow.AddMilliseconds(ShutdownWaitMilliseconds);
            while (owned.Any(record => !record.Retired) && DateTime.UtcNow < limit && !cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(ShutdownPollMilliseconds).ConfigureAwait(false);
            }
        }

        /// <inheritdoc />
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            try
            {
                ShutdownAsync(CancellationToken.None).GetAwaiter().GetResult();
            }
            catch (Exception error)
            {
                Log("shutdown during dispose failed: " + error.Message);
            }

            _timer.Dispose();
        }

        private static bool IsLive(InstanceRecord record) =>
            record.State == InstanceState.Starting || record.State == InstanceState.Ready;

        private static bool IsBusyLocked(InstanceRecord record) => record.Stuck || record.Worker.HasPending;

        private InstanceSnapshot SnapshotLocked(InstanceRecord record) =>
            new InstanceSnapshot(
                record.Id,
                record.Mode,
                record.Owned,
                record.Year,
                record.Revision,
                record.ProcessId,
                record.Visible,
                IsBusyLocked(record),
                0);

        private InstanceSnapshot CompleteRun(InstanceRecord record)
        {
            lock (_lock)
            {
                record.LastUsedMs = _clock.ElapsedMilliseconds;
                return SnapshotLocked(record);
            }
        }

        /// <summary>The recovery probe: the session answers a short read on the instance's own thread.</summary>
        private bool ProbeOnWorker(InstanceRecord record)
        {
            var session = record.Session;
            if (session == null)
            {
                throw HostError.Coded(ErrorCodes.SolidworksNotRunning, $"SolidWorks instance {record.Id} is not ready.");
            }

            session.GetInstanceFacts();
            return true;
        }

        /// <summary>
        /// True when the pid still belongs to the process the record was made for: same pid and same start time. A process
        /// that has gone, or whose start time cannot be read, is not treated as the same one (it is not killed).
        /// </summary>
        private static bool SameProcessAsRecord(InstanceRecord record)
        {
            try
            {
                var recorded = record.ProcessObject.StartTime.ToUniversalTime();
                var current = Process.GetProcessById(record.ProcessId).StartTime.ToUniversalTime();
                return recorded == current;
            }
            catch (Exception error) when (error is ArgumentException || error is InvalidOperationException || error is Win32Exception)
            {
                return false;
            }
        }

        /// <summary>Runs on the worker after a call's body. Clears the busy mark a timed-out attached call left behind.</summary>
        private void ClearStuck(InstanceRecord record, CallMarker call)
        {
            lock (_lock)
            {
                call.Finished = true;
                record.Stuck = false;
            }
        }

        private async Task<InstanceRecord> ResolveAsync(InstanceTarget target, RequestContext context, Deadline deadline, CancellationToken cancellationToken)
        {
            switch (target.Kind)
            {
                case InstanceTargetKind.Specific:
                {
                    var record = FindForCall(target.InstanceId ?? string.Empty, context);
                    await EnsureAttachReadyAsync(record, AttachWaitEnd(deadline), cancellationToken).ConfigureAwait(false);
                    RequireVersionMatch(record, context);
                    return record;
                }

                case InstanceTargetKind.StartNew:
                    return await StartBackgroundAsync(context, deadline, target.RetireInstanceId, cancellationToken).ConfigureAwait(false);

                case InstanceTargetKind.Attach:
                    return await AttachAsync(context, deadline, cancellationToken).ConfigureAwait(false);

                default:
                    throw new ArgumentOutOfRangeException(nameof(target));
            }
        }

        /// <summary>
        /// A specific instance must be live and of the request's instance mode. An instance of another mode is treated as
        /// not found, so a request never reaches the other mode's documents. The version pin is checked after readiness.
        /// </summary>
        private InstanceRecord FindForCall(string instanceId, RequestContext context)
        {
            lock (_lock)
            {
                if (!_records.TryGetValue(instanceId, out var record) || !IsLive(record))
                {
                    throw HostError.Coded(
                        ErrorCodes.SolidworksNotRunning,
                        $"No SolidWorks instance {instanceId} is running. It may have been closed; call documents.list.");
                }

                if (record.Mode != context.InstanceMode)
                {
                    throw HostError.Coded(
                        ErrorCodes.SolidworksNotRunning,
                        $"SolidWorks instance {instanceId} runs in {record.Mode} mode, so this {context.InstanceMode} mode request cannot use it. Call documents.list to see the instances of this mode.");
                }

                return record;
            }
        }

        private void RequireVersionMatch(InstanceRecord record, RequestContext context)
        {
            int? year;
            lock (_lock)
            {
                year = record.Year;
            }

            var pin = context.SolidworksVersion;
            if (!SolidWorksVersions.MatchesPin(pin, year))
            {
                throw HostError.Coded(
                    ErrorCodes.VersionMismatch,
                    $"Instance {record.Id} runs SolidWorks {VersionText(year)}, but the pinned version is {pin}.",
                    new JObject
                    {
                        ["pinned_version"] = WireJson.Value(pin),
                        ["running_versions"] = new JArray(WireJson.Value(year)),
                    });
            }
        }

        /// <summary>Latest time (monotonic ms) a caller waits for attach readiness: now + min(3 s, remaining budget).</summary>
        private long AttachWaitEnd(Deadline deadline) =>
            _clock.ElapsedMilliseconds + Math.Min(AttachReadyMilliseconds, (long)deadline.Remaining.TotalMilliseconds);

        /// <summary>
        /// Waits, at most until <paramref name="endMs"/>, for an attached instance's readiness operation. Returns when the
        /// instance is ready. Otherwise throws: the not-answering error while the operation is still outstanding (it stays
        /// remembered until it finishes), solidworks_not_running when it has finished without a ready instance.
        /// Owned instances are not waited for here.
        /// </summary>
        private async Task EnsureAttachReadyAsync(InstanceRecord record, long endMs, CancellationToken cancellationToken)
        {
            if (record.Owned)
            {
                return;
            }

            await WaitForReadinessAsync(new[] { record }, endMs, cancellationToken).ConfigureAwait(false);

            InstanceState state;
            lock (_lock)
            {
                state = record.State;
            }

            if (state == InstanceState.Ready)
            {
                return;
            }

            if (state == InstanceState.Starting)
            {
                Log($"attach {record.Id}: still not answering (pid {record.ProcessId}); the readiness operation stays outstanding");
                throw HostError.AttachNotReady(record.Id, record.ProcessId);
            }

            throw HostError.Coded(
                ErrorCodes.SolidworksNotRunning,
                $"SolidWorks instance {record.Id} stopped before it became ready. Call documents.list to see the current instances.");
        }

        private async Task<InstanceRecord> StartBackgroundAsync(RequestContext context, Deadline deadline, string? retireInstanceId, CancellationToken cancellationToken)
        {
            var install = ChooseInstall(SolidWorksInstallations.Discover(), context.SolidworksVersion);

            if (retireInstanceId != null)
            {
                await RetireForRoomAsync(retireInstanceId, context, deadline, cancellationToken).ConfigureAwait(false);
            }

            lock (_lock)
            {
                EnsureAcceptingLocked();
                var owned = _records.Values.Count(record =>
                    record.Owned && record.Mode == ProtocolInfo.InstanceModes.Background && record.State != InstanceState.Stopped);
                if (owned + _launching >= context.MaxInstances)
                {
                    throw HostError.Coded(
                        ErrorCodes.InstanceLimit,
                        $"All {context.MaxInstances} background SolidWorks instances are in use. Try again in a moment.",
                        new JObject { ["max_instances"] = WireJson.Value(context.MaxInstances) });
                }

                _launching++;
            }

            Process process;
            try
            {
                process = ProcessControl.Launch(install.ExePath);
            }
            catch (Exception error) when (error is Win32Exception || error is InvalidOperationException)
            {
                lock (_lock)
                {
                    _launching--;
                }

                Log($"start failed: SolidWorks {install.Year} could not be launched ({error.GetType().Name})");
                throw HostError.Coded(
                    ErrorCodes.SolidworksStartFailed,
                    $"Windows could not start SolidWorks {install.Year}.",
                    new JObject { ["version"] = WireJson.Value(install.Year) });
            }

            InstanceRecord? created = null;
            lock (_lock)
            {
                _launching--;
                if (!_shuttingDown)
                {
                    created = NewRecordLocked(ProtocolInfo.InstanceModes.Background, true, install.Year, process);
                    created.IdleShutdownSeconds = context.IdleShutdownSeconds;
                }
            }

            if (created == null)
            {
                RunDetached(async () =>
                {
                    await ProcessControl.KillAsync(process, Log).ConfigureAwait(false);
                    process.Dispose();
                });
                throw HostError.Coded(ErrorCodes.SolidworksNotRunning, "The SolidWorks agent is shutting down.");
            }

            var record = created;
            Log($"start {record.Id}: SolidWorks {install.Year} pid {record.ProcessId} from {install.ExePath}");

            var startup = new WorkItem<bool>(
                () => InstanceStartup.RunOwned(record, deadline, _sessionFactory, _clock, Log),
                null);
            if (!record.Worker.TryEnqueue(startup))
            {
                throw HostError.Coded(ErrorCodes.SolidworksStartFailed, "The SolidWorks worker stopped before startup.");
            }

            _ = startup.Task.ContinueWith(
                task =>
                {
                    if (task.IsFaulted)
                    {
                        RunDetached(() => KillAndFinishAsync(record, "startup failed"));
                    }
                },
                TaskContinuationOptions.ExecuteSynchronously);

            var outcome = await DeadlineWait.WaitAsync(startup.Task, deadline, cancellationToken).ConfigureAwait(false);
            switch (outcome)
            {
                case WaitOutcome.Completed:
                    try
                    {
                        await startup.Task.ConfigureAwait(false);
                    }
                    catch (Exception)
                    {
                        await KillAndFinishAsync(record, "startup failed").ConfigureAwait(false);
                        throw;
                    }

                    return record;

                case WaitOutcome.TimedOut:
                    Log($"start {record.Id}: not ready within the deadline; killing pid {record.ProcessId}");
                    await KillAndFinishAsync(record, "not ready within the deadline").ConfigureAwait(false);
                    throw HostError.Coded(
                        ErrorCodes.InstanceTimeout,
                        $"SolidWorks {install.Year} did not become ready within {(int)deadline.Total.TotalSeconds} seconds.",
                        new JObject { ["instance_id"] = WireJson.Value(record.Id) });

                default:
                    // The caller gave up; the launch continues and the instance becomes ready for later calls.
                    throw new OperationCanceledException(cancellationToken);
            }
        }

        /// <summary>
        /// Makes room for a launch (F6): closes the idle other-year instance Core chose, through the normal close path
        /// (so the unsaved-work rule applies). The close counts against the request deadline. The launch decision that
        /// follows counts the instances again.
        /// </summary>
        private async Task RetireForRoomAsync(string instanceId, RequestContext context, Deadline deadline, CancellationToken cancellationToken)
        {
            InstanceRecord record;
            lock (_lock)
            {
                if (!_records.TryGetValue(instanceId, out var found) || !IsLive(found))
                {
                    return;
                }

                // Core checked the document count; the rule here checks what the host tracks itself.
                if (!InstanceRoom.CanRetire(SnapshotLocked(found), context.SolidworksVersion))
                {
                    throw HostError.Coded(
                        ErrorCodes.InstanceLimit,
                        $"SolidWorks instance {instanceId} cannot be closed to make room right now. Try again in a moment.",
                        new JObject { ["max_instances"] = WireJson.Value(context.MaxInstances) });
                }

                found.State = InstanceState.Closing;
                record = found;
            }

            Log($"room {record.Id}: closing an idle instance of another pinned version to make room");
            var close = CloseAndFinishAsync(record, "making room for a new instance");
            var outcome = await DeadlineWait.WaitAsync(close, deadline, cancellationToken).ConfigureAwait(false);
            switch (outcome)
            {
                case WaitOutcome.Cancelled:
                    throw new OperationCanceledException(cancellationToken);

                case WaitOutcome.TimedOut:
                    throw HostError.Coded(
                        ErrorCodes.InstanceTimeout,
                        $"SolidWorks instance {record.Id} is still closing to make room. Try again in a moment.",
                        new JObject { ["instance_id"] = WireJson.Value(record.Id) });

                default:
                    await close.ConfigureAwait(false);
                    return;
            }
        }

        /// <summary>
        /// Attach: one readiness operation per running SolidWorks process (created on first sight, reused while it
        /// lives). Each candidate is waited for at most until the attach budget ends; blocked operations stay
        /// outstanding. Ready candidates are chosen by <see cref="AttachSelection"/>.
        /// </summary>
        private async Task<InstanceRecord> AttachAsync(RequestContext context, Deadline deadline, CancellationToken cancellationToken)
        {
            var end = AttachWaitEnd(deadline);
            var operations = new List<InstanceRecord>();
            lock (_lock)
            {
                EnsureAcceptingLocked();
                foreach (var processId in ProcessControl.ListSolidWorksProcessIds(ProcessControl.CurrentSessionId()))
                {
                    if (IsOwnedProcess(processId))
                    {
                        continue;
                    }

                    var existing = FindAttachedLocked(processId);
                    if (existing != null)
                    {
                        operations.Add(existing);
                        continue;
                    }

                    var process = TryOpenAttachedProcess(processId);
                    if (process == null)
                    {
                        continue;
                    }

                    var record = NewRecordLocked(ProtocolInfo.InstanceModes.Attach, false, null, process);
                    record.IdleShutdownSeconds = context.IdleShutdownSeconds;
                    record.Startup = StartReadinessLocked(record);
                    Log($"attach {record.Id}: readiness started for pid {record.ProcessId}");
                    operations.Add(record);
                }
            }

            await WaitForReadinessAsync(operations, end, cancellationToken).ConfigureAwait(false);
            return SelectAttached(operations, context);
        }

        /// <summary>
        /// Waits, in order, for the readiness operations of the attached records, never past <paramref name="endMs"/>.
        /// Operations that are still running at the end stay outstanding; nothing is stopped or started here.
        /// </summary>
        private async Task WaitForReadinessAsync(IEnumerable<InstanceRecord> records, long endMs, CancellationToken cancellationToken)
        {
            foreach (var record in records)
            {
                var operation = record.Startup;
                if (record.Owned || operation == null || operation.IsCompleted)
                {
                    continue;
                }

                var remaining = TimeSpan.FromMilliseconds(Math.Max(0, endMs - _clock.ElapsedMilliseconds));
                var outcome = await DeadlineWait.WaitAsync(operation, remaining, cancellationToken).ConfigureAwait(false);
                if (outcome == WaitOutcome.Cancelled)
                {
                    throw new OperationCanceledException(cancellationToken);
                }
            }
        }

        /// <summary>Classifies the attach operations after the bounded wait and picks the instance to use.</summary>
        private InstanceRecord SelectAttached(List<InstanceRecord> operations, RequestContext context)
        {
            var candidates = new List<AttachCandidate>();
            var blocked = new List<InstanceRecord>();
            var failure = (Exception?)null;
            lock (_lock)
            {
                foreach (var record in operations)
                {
                    if (record.State == InstanceState.Ready)
                    {
                        candidates.Add(new AttachCandidate(record.ProcessId, record.Year));
                    }
                    else if (record.State == InstanceState.Starting)
                    {
                        blocked.Add(record);
                    }
                    else
                    {
                        failure ??= record.Startup?.Exception?.InnerException;
                    }
                }
            }

            if (candidates.Count == 0 && blocked.Count > 0)
            {
                var first = blocked[0];
                throw HostError.AttachNotReady(first.Id, first.ProcessId);
            }

            if (candidates.Count == 0 && failure is HostErrorException hostError)
            {
                throw hostError;
            }

            var chosen = AttachSelection.Choose(candidates, context.SolidworksVersion);
            return operations.First(record => record.ProcessId == chosen.ProcessId && record.State == InstanceState.Ready);
        }

        /// <summary>
        /// The readiness operation of an attached process: runs on the record's STA worker until the instance is ready or
        /// the process goes away. A failure retires the record (forgotten); success makes it ready for later calls.
        /// </summary>
        private Task<bool> StartReadinessLocked(InstanceRecord record)
        {
            var readiness = new WorkItem<bool>(
                () => InstanceStartup.RunAttach(record, _sessionFactory, _clock, Log),
                null);
            if (!record.Worker.TryEnqueue(readiness))
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "The SolidWorks worker stopped before attach.");
            }

            _ = readiness.Task.ContinueWith(
                task =>
                {
                    if (task.IsFaulted)
                    {
                        Finish(record, "attach readiness failed");
                    }
                },
                TaskContinuationOptions.ExecuteSynchronously);

            return readiness.Task;
        }

        private void EnqueueOnRecord<T>(InstanceRecord record, WorkItem<T> item, RequestContext context)
        {
            lock (_lock)
            {
                if (!IsLive(record))
                {
                    throw HostError.Coded(
                        ErrorCodes.SolidworksNotRunning,
                        $"SolidWorks instance {record.Id} is no longer running. Call documents.list to see the current instances.");
                }

                if (record.Stuck)
                {
                    throw HostError.Coded(
                        ErrorCodes.InstanceTimeout,
                        $"SolidWorks instance {record.Id} is still busy with a call that passed its deadline. Try again later.",
                        new JObject { ["instance_id"] = WireJson.Value(record.Id) });
                }

                if (!record.Worker.TryEnqueue(item))
                {
                    throw HostError.Coded(ErrorCodes.SolidworksNotRunning, $"SolidWorks instance {record.Id} is stopping.");
                }

                record.LastUsedMs = _clock.ElapsedMilliseconds;
                record.IdleShutdownSeconds = context.IdleShutdownSeconds;
            }
        }

        private T RunOnWorker<T>(InstanceRecord record, Func<ISolidWorksSession, T> work)
        {
            try
            {
                var session = record.Session;
                if (session == null)
                {
                    throw HostError.Coded(ErrorCodes.SolidworksNotRunning, $"SolidWorks instance {record.Id} is not ready.");
                }

                return work(session);
            }
            catch (Exception error) when (!(error is HostErrorException) && ProcessControl.IsGone(record.ProcessObject))
            {
                throw HostError.Coded(
                    ErrorCodes.SolidworksNotRunning,
                    $"SolidWorks instance {record.Id} exited while the request was running.",
                    new JObject { ["exception_type"] = WireJson.Value(error.GetType().Name) });
            }
            finally
            {
                RefreshVisible(record);
            }
        }

        /// <summary>
        /// Runs on the STA thread after every call. An owned instance has its visible titled windows hidden again; the
        /// reported visibility is the window check after that. An attached instance is only read.
        /// </summary>
        private void RefreshVisible(InstanceRecord record)
        {
            var application = record.App;
            if (application == null)
            {
                return;
            }

            bool? visible = record.Owned
                ? InstanceStartup.EnsureHidden(record, application, Log)
                : InstanceStartup.ReadVisible(application);
            if (visible.HasValue)
            {
                lock (_lock)
                {
                    record.Visible = visible.Value;
                }
            }
        }

        private async Task<HostErrorException> TimeoutErrorAsync<T>(InstanceRecord record, WorkItem<T> item, CallMarker call, Deadline deadline)
        {
            var seconds = (int)deadline.Total.TotalSeconds;
            if (item.TryAbandon())
            {
                Log($"{record.Id}: call timed out while queued; nothing was stopped");
                return HostError.Coded(
                    ErrorCodes.InstanceTimeout,
                    $"SolidWorks instance {record.Id} was busy with another call for the whole {seconds}-second deadline. Nothing was stopped; try again.",
                    new JObject { ["instance_id"] = WireJson.Value(record.Id) });
            }

            if (record.Owned)
            {
                Log($"{record.Id}: deadline passed during a call; killing the host-launched process");
                await KillAndFinishAsync(record, "deadline passed during a call").ConfigureAwait(false);
                return HostError.Coded(
                    ErrorCodes.OperationTimeout,
                    $"The request passed its {seconds}-second deadline. SolidWorks instance {record.Id} was stopped; the next request starts a new one.",
                    new JObject
                    {
                        ["instance_id"] = WireJson.Value(record.Id),
                        ["instance_restarted"] = WireJson.Value(true),
                    });
            }

            lock (_lock)
            {
                // Set only while the call has not finished; ClearStuck takes the same lock, so the order is safe.
                if (!call.Finished)
                {
                    record.Stuck = true;
                }
            }

            Log($"{record.Id}: deadline passed during a call on an attached instance; marked busy, not stopped");
            return HostError.Coded(
                ErrorCodes.OperationTimeout,
                $"The request passed its {seconds}-second deadline. SolidWorks instance {record.Id} is still busy; further requests to it fail until the call returns.",
                new JObject
                {
                    ["instance_id"] = WireJson.Value(record.Id),
                    ["instance_restarted"] = WireJson.Value(false),
                });
        }

        /// <summary>
        /// Graceful exit of an owned instance (idle shutdown, host shutdown, or making room). Queued calls fail first.
        /// The save state is decided by <see cref="CloseDecision"/>: known-clean closes; known-unsaved releases; unknown
        /// closes only when no titled window was ever seen after the first hide, otherwise releases. A release stops
        /// tracking the instance and leaves the process running (no close, no kill). A close is graceful; the process is
        /// killed only after the grace period, and only on this path.
        /// </summary>
        private async Task CloseAndFinishAsync(InstanceRecord record, string reason)
        {
            record.Worker.FailQueued(HostError.Coded(
                ErrorCodes.SolidworksNotRunning,
                $"SolidWorks instance {record.Id} is closing ({reason})."));

            InstanceStartup.NoteWindowsBeforeClose(record);
            var check = await CheckSaveStateAsync(record).ConfigureAwait(false);
            var everVisible = record.EverVisibleAfterHide;
            var mayClose = CloseDecision.MayClose(check.State, everVisible);
            Log($"close {record.Id} ({reason}): {(mayClose ? "closing" : "releasing without close or kill")}; save state {DescribeSaveCheck(check)}; window seen visible after hide: {(everVisible ? "yes" : "no")}");

            if (!mayClose)
            {
                Finish(record, $"{DescribeSaveCheck(check)}; process left running");
                return;
            }

            await CloseGracefullyAsync(record, reason).ConfigureAwait(false);
        }

        private static string DescribeSaveCheck(SaveCheck check)
        {
            switch (check.State)
            {
                case SaveState.Clean:
                    return "clean";
                case SaveState.Unsaved:
                    return $"unsaved changes ({string.Join("; ", check.Unsaved)})";
                default:
                    return $"unknown ({check.Reason})";
            }
        }

        /// <summary>
        /// Runs the save-state check on the STA thread, waiting at most <see cref="SaveCheckMilliseconds"/>. A check
        /// that fails or does not finish in time is unknown.
        /// </summary>
        private async Task<SaveCheck> CheckSaveStateAsync(InstanceRecord record)
        {
            var item = new WorkItem<SaveCheck>(() => CheckSaveStateOnSta(record), null);
            if (!record.Worker.TryEnqueue(item))
            {
                return SaveCheck.Unknown("the instance worker had stopped");
            }

            var first = await Task.WhenAny(item.Task, Task.Delay(SaveCheckMilliseconds)).ConfigureAwait(false);
            if (!ReferenceEquals(first, item.Task))
            {
                return SaveCheck.Unknown($"the document check did not finish within {SaveCheckMilliseconds} ms");
            }

            try
            {
                return await item.Task.ConfigureAwait(false);
            }
            catch (Exception error)
            {
                return SaveCheck.Unknown($"the document check failed ({error.GetType().Name})");
            }
        }

        private SaveCheck CheckSaveStateOnSta(InstanceRecord record)
        {
            var unsaved = InstanceStartup.UnsavedDocumentPaths(record, Log);
            if (unsaved == null)
            {
                return SaveCheck.Unknown("the open documents could not be listed");
            }

            return unsaved.Count > 0 ? SaveCheck.WithUnsaved(unsaved) : SaveCheck.Clean();
        }

        /// <summary>Closes without saving on the STA thread, then waits for the exit; kills only after the grace period.</summary>
        private async Task CloseGracefullyAsync(InstanceRecord record, string reason)
        {
            var close = new WorkItem<bool>(() =>
            {
                InstanceStartup.CloseAndExit(record, Log);
                return true;
            }, null);
            if (!record.Worker.TryEnqueue(close))
            {
                Log($"close {record.Id}: the worker had already stopped; killing pid {record.ProcessId}");
                await KillAndFinishAsync(record, reason).ConfigureAwait(false);
                return;
            }

            var watch = Stopwatch.StartNew();
            var first = await Task.WhenAny(close.Task, Task.Delay(CloseGraceMilliseconds)).ConfigureAwait(false);
            if (!ReferenceEquals(first, close.Task))
            {
                Log($"close {record.Id}: SolidWorks did not finish closing within {CloseGraceMilliseconds} ms");
                await KillAndFinishAsync(record, reason).ConfigureAwait(false);
                return;
            }

            try
            {
                await close.Task.ConfigureAwait(false);
            }
            catch (Exception error)
            {
                Log($"close {record.Id}: unexpected failure ({error.GetType().Name}); the process is left running");
                Finish(record, "close failed; process left running");
                return;
            }

            var exited = await ProcessControl.WaitForExitAsync(record.ProcessObject, CloseGraceMilliseconds).ConfigureAwait(false);
            if (!exited)
            {
                Log($"close {record.Id}: pid {record.ProcessId} still running {CloseGraceMilliseconds} ms after exit was requested");
                await KillAndFinishAsync(record, reason).ConfigureAwait(false);
                return;
            }

            Log($"close {record.Id}: pid {record.ProcessId} exited {watch.ElapsedMilliseconds} ms after the close started");
            Finish(record, reason);
        }

        private async Task KillAndFinishAsync(InstanceRecord record, string reason)
        {
            if (!Retire(record))
            {
                return;
            }

            Log($"kill {record.Id} pid {record.ProcessId}: {reason}");
            var watch = Stopwatch.StartNew();
            await ProcessControl.KillAsync(record.ProcessObject, Log).ConfigureAwait(false);
            Log($"kill {record.Id} pid {record.ProcessId}: killed after {watch.ElapsedMilliseconds} ms");
            FinishRetired(record, reason);
        }

        private void Finish(InstanceRecord record, string reason)
        {
            if (Retire(record))
            {
                FinishRetired(record, reason);
            }
        }

        /// <summary>Moves the record to Stopped and removes it from the registry. True only for the first caller.</summary>
        private bool Retire(InstanceRecord record)
        {
            lock (_lock)
            {
                if (record.State == InstanceState.Stopped)
                {
                    return false;
                }

                record.State = InstanceState.Stopped;
                record.MarkRetired();
                _records.Remove(record.Id);
                return true;
            }
        }

        private void FinishRetired(InstanceRecord record, string reason)
        {
            record.Worker.Stop(HostError.Coded(
                ErrorCodes.SolidworksNotRunning,
                $"SolidWorks instance {record.Id} stopped ({reason})."));
            Log($"exit {record.Id} ({reason}), pid {record.ProcessId}");
            DisposeProcess(record);
            RaiseInstanceStopped(record.Id);
        }

        /// <summary>Disposes the record's Process object once the record is retired. Its id and exit state are no longer used.</summary>
        private void DisposeProcess(InstanceRecord record)
        {
            try
            {
                record.ProcessObject.Dispose();
            }
            catch (Exception error)
            {
                Log($"{record.Id}: disposing the process object failed ({error.GetType().Name})");
            }
        }

        private void RaiseInstanceStopped(string instanceId)
        {
            var handler = InstanceStopped;
            if (handler == null)
            {
                return;
            }

            try
            {
                handler(instanceId);
            }
            catch (Exception error)
            {
                Log($"InstanceStopped handler failed for {instanceId}: {error.Message}");
            }
        }

        private void OnTick(object? state)
        {
            if (_disposed)
            {
                return;
            }

            if (Interlocked.CompareExchange(ref _ticking, 1, 0) != 0)
            {
                return;
            }

            try
            {
                ReapExitedInstances();
                CheckIdleInstances();
            }
            catch (Exception error)
            {
                Log("runtime watchdog failed: " + error.Message);
            }
            finally
            {
                Interlocked.Exchange(ref _ticking, 0);
            }
        }

        private void ReapExitedInstances()
        {
            List<InstanceRecord> snapshot;
            lock (_lock)
            {
                snapshot = _records.Values.ToList();
            }

            foreach (var record in snapshot)
            {
                if (ProcessControl.IsGone(record.ProcessObject))
                {
                    Finish(record, "process exited");
                }
            }
        }

        private void CheckIdleInstances()
        {
            var now = _clock.ElapsedMilliseconds;
            if (now - _lastIdleCheckMs < IdleCheckMilliseconds)
            {
                return;
            }

            _lastIdleCheckMs = now;
            var idle = new List<InstanceRecord>();
            lock (_lock)
            {
                foreach (var record in _records.Values)
                {
                    if (record.State != InstanceState.Ready)
                    {
                        continue;
                    }

                    var idleFor = TimeSpan.FromMilliseconds(now - record.LastUsedMs);
                    if (IdlePolicy.IsEligibleForIdleShutdown(record.Mode, record.Owned, IsBusyLocked(record), idleFor, record.IdleShutdownSeconds))
                    {
                        record.State = InstanceState.Closing;
                        idle.Add(record);
                    }
                }
            }

            foreach (var record in idle)
            {
                Log($"idle shutdown {record.Id}: no call for {record.IdleShutdownSeconds} s");
                RunDetached(() => CloseAndFinishAsync(record, "idle shutdown"));
            }
        }

        private InstanceRecord NewRecordLocked(string mode, bool owned, int? year, Process process)
        {
            var record = new InstanceRecord(++_nextNumber, mode, owned, year, process, _lock);
            record.Worker.ExitCallback = () => InstanceStartup.ReleaseApplication(record);
            record.LastUsedMs = _clock.ElapsedMilliseconds;
            _records.Add(record.Id, record);
            record.Worker.Start();
            return record;
        }

        private InstanceRecord? FindAttachedLocked(int processId) =>
            _records.Values.FirstOrDefault(record =>
                !record.Owned && record.ProcessId == processId && IsLive(record));

        private bool IsOwnedProcess(int processId)
        {
            lock (_lock)
            {
                return _records.Values.Any(record => record.Owned && record.ProcessId == processId && record.State != InstanceState.Stopped);
            }
        }

        /// <summary>The process object of a running SolidWorks, or null when it has already gone.</summary>
        private static Process? TryOpenAttachedProcess(int processId)
        {
            try
            {
                return Process.GetProcessById(processId);
            }
            catch (Exception error) when (error is ArgumentException || error is InvalidOperationException || error is Win32Exception)
            {
                return null;
            }
        }

        private void EnsureAcceptingLocked()
        {
            if (_shuttingDown || _disposed)
            {
                throw HostError.Coded(ErrorCodes.SolidworksNotRunning, "The SolidWorks agent is shutting down.");
            }
        }

        private static InstalledSolidWorks ChooseInstall(IReadOnlyList<InstalledSolidWorks> installs, int? pin)
        {
            if (installs.Count == 0)
            {
                throw HostError.Coded(
                    ErrorCodes.VersionNotInstalled,
                    "No supported SolidWorks (2022 to 2035) was found on this computer.");
            }

            if (!pin.HasValue)
            {
                return installs[0];
            }

            var match = installs.FirstOrDefault(install => install.Year == pin.Value);
            if (match == null)
            {
                throw HostError.Coded(
                    ErrorCodes.VersionNotInstalled,
                    $"SolidWorks {pin.Value} is not installed on this computer.",
                    new JObject
                    {
                        ["pinned_version"] = WireJson.Value(pin.Value),
                        ["installed_versions"] = new JArray(installs.Select(install => WireJson.Value(install.Year)).ToArray()),
                    });
            }

            return match;
        }

        private static string VersionText(int? year) =>
            year.HasValue ? year.Value.ToString(CultureInfo.InvariantCulture) : "unknown";

        private void RunDetached(Func<Task> work)
        {
            Task.Run(async () =>
            {
                try
                {
                    await work().ConfigureAwait(false);
                }
                catch (Exception error)
                {
                    Log("background instance task failed: " + error.Message);
                }
            });
        }

        private void Log(string message)
        {
            try
            {
                _log(message);
            }
            catch (Exception)
            {
                // A failing log sink must not break the instance lifecycle.
            }
        }

        /// <summary>Whether one call's body has finished. Guarded by the manager lock.</summary>
        private sealed class CallMarker
        {
            public bool Finished { get; set; }
        }
    }
}
