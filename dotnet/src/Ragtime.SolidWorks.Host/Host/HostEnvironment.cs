using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Principal;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Identity, names and paths of this host process for the current user, resolved once per command.</summary>
    internal sealed class HostEnvironment
    {
        /// <summary>File name of the host executable.</summary>
        public const string ExeName = "ragtime-solidworks-host.exe";

        private HostEnvironment(string accountName, string sid, string exePath, string hostVersion, HostLog log)
        {
            AccountName = accountName;
            Sid = sid;
            ExePath = exePath;
            HostVersion = hostVersion;
            Log = log;
        }

        /// <summary>DOMAIN\user of the current account.</summary>
        public string AccountName { get; }

        /// <summary>SID of the current account; every name below is derived from it.</summary>
        public string Sid { get; }

        /// <summary>Full path of this executable.</summary>
        public string ExePath { get; }

        /// <summary>The only folder install registers the task from: %LOCALAPPDATA%\RagtimeSolidWorks\host.</summary>
        public static string CanonicalInstallDirectory =>
            Path.GetFullPath(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "RagtimeSolidWorks", "host"));

        /// <summary>The executable path the logon task runs: the canonical install location.</summary>
        public static string CanonicalExePath => Path.Combine(CanonicalInstallDirectory, ExeName);

        /// <summary>True when this executable runs from the canonical install location (case-insensitive full path).</summary>
        public bool RunsFromCanonicalInstall =>
            string.Equals(Path.GetFullPath(ExePath), CanonicalExePath, StringComparison.OrdinalIgnoreCase);

        /// <summary>Assembly informational version without any "+metadata" suffix.</summary>
        public string HostVersion { get; }

        /// <summary>Host log for this user.</summary>
        public HostLog Log { get; }

        /// <summary>Named pipe name without the \\.\pipe\ prefix.</summary>
        public string PipeName => HostNames.PipeName(Sid);

        /// <summary>Logon scheduled task name.</summary>
        public string TaskName => HostNames.TaskName(Sid);

        /// <summary>Global single-instance mutex name.</summary>
        public string MutexName => HostNames.MutexName(Sid);

        /// <summary>Resolves the environment of the current process.</summary>
        public static HostEnvironment Create(HostLog log)
        {
            using (var identity = WindowsIdentity.GetCurrent())
            {
                var exePath = Process.GetCurrentProcess().MainModule.FileName;
                return new HostEnvironment(identity.Name, identity.User.Value, exePath, ReadHostVersion(), log);
            }
        }

        /// <summary>The host version from the assembly informational version (contract section 3).</summary>
        public static string ReadHostVersion()
        {
            var assembly = typeof(HostEnvironment).Assembly;
            var informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
                ?? assembly.GetName().Version.ToString(3);
            var plus = informational.IndexOf('+');
            return plus >= 0 ? informational.Substring(0, plus) : informational;
        }
    }
}
