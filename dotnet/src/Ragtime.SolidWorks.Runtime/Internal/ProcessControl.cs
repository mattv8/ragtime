using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>Process helpers: launch, exit checks, kill, and the session's SLDWORKS processes.</summary>
    internal static class ProcessControl
    {
        private const string SolidWorksProcessName = "SLDWORKS";
        private const int KillWaitMilliseconds = 5000;

        /// <summary>Starts SLDWORKS.exe with the /r switch (a fresh instance). Throws on failure.</summary>
        public static Process Launch(string exePath)
        {
            var info = new ProcessStartInfo
            {
                FileName = exePath,
                Arguments = "/r",
                UseShellExecute = false,
                WorkingDirectory = Path.GetDirectoryName(exePath) ?? string.Empty,
            };

            return Process.Start(info) ?? throw new InvalidOperationException("Windows did not start the SolidWorks process.");
        }

        /// <summary>True when the process has exited. An unreadable state counts as running.</summary>
        public static bool IsGone(Process process)
        {
            try
            {
                return process.HasExited;
            }
            catch (InvalidOperationException)
            {
                return true;
            }
            catch (Win32Exception)
            {
                return false;
            }
        }

        /// <summary>Waits off the calling thread for the process to exit. True when it exited.</summary>
        public static Task<bool> WaitForExitAsync(Process process, int milliseconds) =>
            Task.Run(() => WaitForExitSync(process, milliseconds));

        /// <summary>Kills the process (if still alive) and waits briefly for it to go away.</summary>
        public static async Task KillAsync(Process process, Action<string> log)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill();
                }
            }
            catch (InvalidOperationException)
            {
                // The process is already gone.
            }
            catch (Win32Exception error)
            {
                log($"kill of pid {process.Id} failed: {error.Message}");
            }

            var gone = await WaitForExitAsync(process, KillWaitMilliseconds).ConfigureAwait(false);
            if (!gone)
            {
                log($"pid {process.Id} is still running after the kill");
            }
        }

        /// <summary>Windows session id of this host process.</summary>
        public static int CurrentSessionId()
        {
            using (var self = Process.GetCurrentProcess())
            {
                return self.SessionId;
            }
        }

        /// <summary>Process ids of SLDWORKS processes in the given Windows session.</summary>
        public static IReadOnlyList<int> ListSolidWorksProcessIds(int sessionId)
        {
            var ids = new List<int>();
            foreach (var process in Process.GetProcessesByName(SolidWorksProcessName))
            {
                try
                {
                    if (process.SessionId == sessionId)
                    {
                        ids.Add(process.Id);
                    }
                }
                catch (InvalidOperationException)
                {
                    // Exited between listing and the query.
                }
                catch (Win32Exception)
                {
                    // Not readable; not a candidate.
                }
                finally
                {
                    process.Dispose();
                }
            }

            return ids;
        }

        private static bool WaitForExitSync(Process process, int milliseconds)
        {
            try
            {
                return process.WaitForExit(milliseconds);
            }
            catch (InvalidOperationException)
            {
                return true;
            }
            catch (Win32Exception)
            {
                return IsGone(process);
            }
        }
    }
}
