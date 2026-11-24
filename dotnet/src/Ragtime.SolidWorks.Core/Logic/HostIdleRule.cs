using System.Collections.Generic;
using System.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Decides whether the host may stop now for <c>host.shutdown</c> with <c>if_idle_seconds</c>. Pure: the caller
    /// passes the facts it read under the activity lock. Reasons come back in the fixed order of the constants below.
    /// </summary>
    public static class HostIdleRule
    {
        /// <summary>At least one work request is running.</summary>
        public const string RequestInFlight = "request_in_flight";

        /// <summary>The host was used less than <c>if_idle_seconds</c> ago.</summary>
        public const string RecentlyUsed = "recently_used";

        /// <summary>An instance is busy (a call runs on it, or a timed-out call has not returned).</summary>
        public const string InstanceBusy = "instance_busy";

        /// <summary>A host-owned background instance has documents open, so stopping would lose them.</summary>
        public const string DocumentsOpen = "documents_open";

        /// <summary>A PDM item whose caller gave up is still running.</summary>
        public const string PdmBusy = "pdm_busy";

        /// <summary>
        /// Returns the reasons the host must not stop, in this order: <see cref="RequestInFlight"/> (inFlight &gt; 0),
        /// <see cref="RecentlyUsed"/> (idleSeconds below ifIdleSeconds), <see cref="InstanceBusy"/> (any busy instance),
        /// <see cref="DocumentsOpen"/> (any owned background instance with documents; attach-mode and non-owned
        /// instances never count), <see cref="PdmBusy"/>. An empty list means it is safe to stop.
        /// </summary>
        public static IReadOnlyList<string> BlockingReasons(
            int inFlight,
            int idleSeconds,
            int ifIdleSeconds,
            IReadOnlyList<InstanceSnapshot> instances,
            bool pdmBusy)
        {
            var reasons = new List<string>();
            if (inFlight > 0)
            {
                reasons.Add(RequestInFlight);
            }

            if (idleSeconds < ifIdleSeconds)
            {
                reasons.Add(RecentlyUsed);
            }

            if (instances.Any(instance => instance.Busy))
            {
                reasons.Add(InstanceBusy);
            }

            if (instances.Any(HasOwnedDocuments))
            {
                reasons.Add(DocumentsOpen);
            }

            if (pdmBusy)
            {
                reasons.Add(PdmBusy);
            }

            return reasons;
        }

        private static bool HasOwnedDocuments(InstanceSnapshot instance) =>
            instance.Mode == ProtocolInfo.InstanceModes.Background && instance.Owned && instance.DocumentCount > 0;
    }
}
