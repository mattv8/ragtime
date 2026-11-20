using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>What <c>document.open</c> should do with the pool.</summary>
    public enum PoolAction
    {
        /// <summary>Launch a new background instance.</summary>
        StartNew,

        /// <summary>Use an existing instance (see <see cref="PoolChoice.InstanceId"/>).</summary>
        UseExisting,

        /// <summary>Every instance is busy and the pool is full: fail with instance_limit.</summary>
        LimitReached,
    }

    /// <summary>Decision of <see cref="InstancePool.ChooseForOpen"/>.</summary>
    public sealed class PoolChoice
    {
        /// <summary>Creates a choice.</summary>
        public PoolChoice(PoolAction action, string? instanceId)
        {
            Action = action;
            InstanceId = instanceId;
        }

        /// <summary>What to do.</summary>
        public PoolAction Action { get; }

        /// <summary>Instance to use for <see cref="PoolAction.UseExisting"/>; otherwise null.</summary>
        public string? InstanceId { get; }
    }

    /// <summary>
    /// Pool rules for background instances (decision D9), as a pure function over snapshots.
    /// <para>
    /// Without a pin, every background instance is a candidate. With a pinned year (C5), only background instances
    /// of that year are candidates: an instance of another year is never used for a document. Such an instance still
    /// counts toward <c>maxInstances</c> while it is busy or holds documents; an idle instance of another year does
    /// not count and does not block starting a new one.
    /// </para>
    /// <para>Rules, in order:</para>
    /// <list type="number">
    /// <item>No candidate and no counted other-year instance: start one. No candidate but the counted ones fill
    /// the pool: <see cref="PoolAction.LimitReached"/>.</item>
    /// <item>Pick the least-loaded candidate: non-busy before busy, then fewest documents, then the earliest listed.</item>
    /// <item>If that candidate is idle (not busy, no documents), use it.</item>
    /// <item>Otherwise, if the candidates plus counted other-year instances are fewer than <c>maxInstances</c>, start a new one.</item>
    /// <item>Otherwise, if every candidate is busy, return <see cref="PoolAction.LimitReached"/> (instance_limit).</item>
    /// <item>Otherwise use the least-loaded candidate (not busy, with documents).</item>
    /// </list>
    /// Attach-mode instances are ignored here.
    /// </summary>
    public static class InstancePool
    {
        /// <summary>Chooses where a new document goes, without a version pin. <paramref name="maxInstances"/> must be at least 1.</summary>
        public static PoolChoice ChooseForOpen(IReadOnlyList<InstanceSnapshot> instances, int maxInstances) =>
            ChooseForOpen(instances, maxInstances, null);

        /// <summary>
        /// Chooses where a new document goes, for the version pin <paramref name="pinnedYear"/> (null: no pin).
        /// <paramref name="maxInstances"/> must be at least 1.
        /// </summary>
        public static PoolChoice ChooseForOpen(IReadOnlyList<InstanceSnapshot> instances, int maxInstances, int? pinnedYear)
        {
            if (maxInstances < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(maxInstances), "The pool needs at least one instance.");
            }

            var background = instances.Where(i => i.Mode == ProtocolInfo.InstanceModes.Background).ToList();
            var candidates = background.Where(i => SolidWorksVersions.MatchesPin(pinnedYear, i.Version)).ToList();
            var counted = background.Count(i => !candidates.Contains(i) && (i.Busy || i.DocumentCount > 0));

            if (candidates.Count == 0)
            {
                return new PoolChoice(counted < maxInstances ? PoolAction.StartNew : PoolAction.LimitReached, null);
            }

            var leastLoaded = candidates
                .OrderBy(i => i.Busy ? 1 : 0)
                .ThenBy(i => i.DocumentCount)
                .First();

            if (!leastLoaded.Busy && leastLoaded.DocumentCount == 0)
            {
                return new PoolChoice(PoolAction.UseExisting, leastLoaded.InstanceId);
            }

            if (candidates.Count + counted < maxInstances)
            {
                return new PoolChoice(PoolAction.StartNew, null);
            }

            if (candidates.All(i => i.Busy))
            {
                return new PoolChoice(PoolAction.LimitReached, null);
            }

            return new PoolChoice(PoolAction.UseExisting, leastLoaded.InstanceId);
        }
    }

    /// <summary>
    /// Making room for a launch (F6). A launch that would exceed <c>max_instances</c> counts every background instance,
    /// including idle ones of another pinned year. Such an instance may be closed first when it is owned by the host,
    /// idle (not busy, no queued work), holds no documents, and runs a known year different from the pin. Pure: the
    /// instance manager calls the same rule to check the instance Core chose.
    /// </summary>
    public static class InstanceRoom
    {
        /// <summary>True when the instance may be closed to make room for a launch pinned to <paramref name="pinnedYear"/>.</summary>
        public static bool CanRetire(InstanceSnapshot instance, int? pinnedYear) =>
            pinnedYear.HasValue
            && instance.Mode == ProtocolInfo.InstanceModes.Background
            && instance.Owned
            && !instance.Busy
            && instance.DocumentCount == 0
            && instance.Version.HasValue
            && instance.Version.Value != pinnedYear.Value;

        /// <summary>The first instance (in listing order) that may be retired, or null when none may.</summary>
        public static InstanceSnapshot? InstanceToRetire(IReadOnlyList<InstanceSnapshot> instances, int? pinnedYear) =>
            instances.FirstOrDefault(instance => CanRetire(instance, pinnedYear));
    }

    /// <summary>A running SolidWorks process seen in the host's session (attach mode).</summary>
    public sealed class AttachCandidate
    {
        /// <summary>Creates a candidate.</summary>
        public AttachCandidate(int processId, int? year)
        {
            ProcessId = processId;
            Year = year;
        }

        /// <summary>Process id.</summary>
        public int ProcessId { get; }

        /// <summary>Release year, or null when unknown.</summary>
        public int? Year { get; }
    }

    /// <summary>Attach-mode selection (contract section 4), a pure function over candidates.</summary>
    public static class AttachSelection
    {
        /// <summary>
        /// Picks the instance to attach to. Candidates are the SolidWorks processes in the host's session.
        /// No candidates: solidworks_not_running. With a pinned year, only candidates of that year qualify;
        /// if candidates exist and none qualify: version_mismatch. Several qualify: the lowest process id.
        /// </summary>
        public static AttachCandidate Choose(IReadOnlyList<AttachCandidate> candidates, int? pinnedYear)
        {
            if (candidates.Count == 0)
            {
                throw HostError.Coded(
                    ErrorCodes.SolidworksNotRunning,
                    "No SolidWorks instance is running in this desktop session. Start SolidWorks, or use background mode.");
            }

            var matching = candidates.Where(c => SolidWorksVersions.MatchesPin(pinnedYear, c.Year)).ToList();
            if (matching.Count == 0)
            {
                var running = new JArray();
                foreach (var candidate in candidates)
                {
                    running.Add(WireJson.Value(candidate.Year));
                }

                throw HostError.Coded(
                    ErrorCodes.VersionMismatch,
                    $"The running SolidWorks is not {pinnedYear}. Start SolidWorks {pinnedYear} or change the pinned version.",
                    new JObject
                    {
                        ["pinned_version"] = WireJson.Value(pinnedYear),
                        ["running_versions"] = running,
                    });
            }

            return matching.OrderBy(c => c.ProcessId).First();
        }
    }

    /// <summary>Idle-shutdown eligibility (background instances only; decision D8).</summary>
    public static class IdlePolicy
    {
        /// <summary>
        /// True when a background instance may be closed by idle shutdown: it is owned by the host, not busy, and
        /// has seen no call for at least <paramref name="idleShutdownSeconds"/>. Open documents do not prevent it.
        /// Attached instances are never eligible.
        /// </summary>
        public static bool IsEligibleForIdleShutdown(string mode, bool owned, bool busy, TimeSpan idleFor, int idleShutdownSeconds)
        {
            return mode == ProtocolInfo.InstanceModes.Background
                && owned
                && !busy
                && idleFor >= TimeSpan.FromSeconds(idleShutdownSeconds);
        }
    }
}
