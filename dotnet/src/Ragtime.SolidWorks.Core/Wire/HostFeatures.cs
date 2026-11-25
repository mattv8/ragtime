using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Feature names the installed host reports in its <c>version</c> capabilities, so Ragtime can tell what an
    /// installed host supports before it relies on a behaviour.
    /// </summary>
    public static class HostFeatures
    {
        /// <summary>The <c>stop</c> command accepts an idle condition (<c>stop --if-idle N</c>).</summary>
        public const string StopIfIdle = "stop-if-idle";

        /// <summary>Every feature this Core build supports, in contract order.</summary>
        public static IReadOnlyList<string> All { get; } = new[] { StopIfIdle };
    }
}
