using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.Win32;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>One installed SolidWorks release.</summary>
    internal sealed class InstalledSolidWorks
    {
        public InstalledSolidWorks(int year, string exePath)
        {
            Year = year;
            ExePath = exePath;
        }

        /// <summary>Release year, e.g. 2024.</summary>
        public int Year { get; }

        /// <summary>Full path of SLDWORKS.exe.</summary>
        public string ExePath { get; }
    }

    /// <summary>
    /// Finds installed SolidWorks releases from the 64-bit registry: ProgIDs SldWorks.Application.&lt;major&gt;,
    /// then the CLSID's LocalServer32 executable. Reads only; starts nothing.
    /// </summary>
    internal static class SolidWorksInstallations
    {
        private const string ProgIdPrefix = "SldWorks.Application.";

        /// <summary>Installed releases in the supported year range, newest first.</summary>
        public static IReadOnlyList<InstalledSolidWorks> Discover()
        {
            var found = new Dictionary<int, InstalledSolidWorks>();
            using (var classesRoot = RegistryKey.OpenBaseKey(RegistryHive.ClassesRoot, RegistryView.Registry64))
            {
                foreach (var name in classesRoot.GetSubKeyNames())
                {
                    if (!name.StartsWith(ProgIdPrefix, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    var majorText = name.Substring(ProgIdPrefix.Length);
                    if (!int.TryParse(majorText, NumberStyles.None, CultureInfo.InvariantCulture, out var major))
                    {
                        continue;
                    }

                    var year = SolidWorksVersions.YearFromMajor(major);
                    if (!SolidWorksVersions.IsSupportedYear(year) || found.ContainsKey(year))
                    {
                        continue;
                    }

                    var exePath = ExecutableForProgId(classesRoot, name);
                    if (exePath != null)
                    {
                        found[year] = new InstalledSolidWorks(year, exePath);
                    }
                }
            }

            return found.Values.OrderByDescending(install => install.Year).ToList();
        }

        /// <summary>Extracts the executable path from a LocalServer32 command line (quotes and arguments removed).</summary>
        public static string? ExecutablePath(string? commandLine)
        {
            if (string.IsNullOrWhiteSpace(commandLine))
            {
                return null;
            }

            var text = commandLine!.Trim();
            if (text.StartsWith("\"", StringComparison.Ordinal))
            {
                var closing = text.IndexOf('"', 1);
                return closing > 1 ? text.Substring(1, closing - 1) : null;
            }

            var exeEnd = text.IndexOf(".exe", StringComparison.OrdinalIgnoreCase);
            return exeEnd >= 0 ? text.Substring(0, exeEnd + 4) : text;
        }

        private static string? ExecutableForProgId(RegistryKey classesRoot, string progId)
        {
            string? clsid;
            using (var clsidKey = classesRoot.OpenSubKey(progId + "\\CLSID"))
            {
                clsid = clsidKey?.GetValue(null) as string;
            }

            if (string.IsNullOrWhiteSpace(clsid))
            {
                return null;
            }

            string? commandLine;
            using (var serverKey = classesRoot.OpenSubKey("CLSID\\" + clsid + "\\LocalServer32"))
            {
                commandLine = serverKey?.GetValue(null) as string;
            }

            var path = ExecutablePath(commandLine);
            return path != null && File.Exists(path) ? path : null;
        }
    }
}
