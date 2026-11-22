using System;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>When a script run folder is old enough for the sweeper to delete it.</summary>
    public static class ScriptRunRetention
    {
        /// <summary>Run folders older than this are deleted by the sweeper.</summary>
        public static readonly TimeSpan MaxAge = TimeSpan.FromHours(24);

        /// <summary>True when a folder created at <paramref name="createdUtc"/> is older than <see cref="MaxAge"/> at <paramref name="nowUtc"/>.</summary>
        public static bool IsExpired(DateTime createdUtc, DateTime nowUtc) => nowUtc - createdUtc > MaxAge;
    }
}
