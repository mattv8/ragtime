using System;
using System.Text.RegularExpressions;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Names and relative paths derived from the user SID (contract section 1).</summary>
    public static class HostNames
    {
        /// <summary>Host executable file name.</summary>
        public const string ExeName = "ragtime-solidworks-host.exe";

        /// <summary>Install directory, relative to %LOCALAPPDATA%.</summary>
        public const string InstallDirRelative = @"RagtimeSolidWorks\host";

        /// <summary>Host log file, relative to %LOCALAPPDATA%.</summary>
        public const string LogRelative = @"RagtimeSolidWorks\logs\host.log";

        /// <summary>Staging directory for uploaded zips, relative to %LOCALAPPDATA%.</summary>
        public const string StagingRelative = @"RagtimeSolidWorks\staging";

        private static readonly Regex SidPattern = new Regex(@"^S-\d+(-\d+)+$", RegexOptions.CultureInvariant);

        /// <summary>Named pipe name (without the \\.\pipe\ prefix) for a user SID.</summary>
        public static string PipeName(string sid) => "ragtime-solidworks-host-" + RequireSid(sid);

        /// <summary>Logon scheduled task name for a user SID.</summary>
        public static string TaskName(string sid) => "RagtimeSolidWorksHost-" + RequireSid(sid);

        /// <summary>Single-instance mutex name for a user SID. Global\ so every session sees it.</summary>
        public static string MutexName(string sid) => @"Global\ragtime-solidworks-host-" + RequireSid(sid);

        /// <summary>Returns the SID unchanged when it has the S-1-... shape; otherwise throws.</summary>
        public static string RequireSid(string sid)
        {
            if (sid == null || !SidPattern.IsMatch(sid))
            {
                throw new ArgumentException("The user SID must look like S-1-5-21-...", nameof(sid));
            }

            return sid;
        }
    }
}
