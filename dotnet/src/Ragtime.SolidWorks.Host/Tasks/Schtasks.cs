using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Outcome of one schtasks.exe run: exit code and combined stdout and stderr on one line.</summary>
    internal sealed class SchtasksResult
    {
        public SchtasksResult(int exitCode, string output)
        {
            ExitCode = exitCode;
            Output = output;
        }

        public int ExitCode { get; }

        public string Output { get; }

        public bool Succeeded => ExitCode == 0;

        /// <summary>The output, clipped to 200 characters, for error messages.</summary>
        public string Summary
        {
            get
            {
                if (Output.Length == 0)
                {
                    return "no output";
                }

                return Output.Length <= 200 ? Output : Output.Substring(0, 200);
            }
        }
    }

    /// <summary>
    /// schtasks.exe wrapper. Arguments go to the process directly (no cmd.exe). Every run is bounded by the caller's
    /// time limit (the child is killed when it expires) and logged with its exit code and output on one line.
    /// </summary>
    internal sealed class Schtasks
    {
        /// <summary>Generous fixed limit for install, stop and uninstall.</summary>
        public static readonly TimeSpan FixedLimit = TimeSpan.FromSeconds(60);

        private readonly HostLog _log;
        private readonly string _executable = Path.Combine(Environment.SystemDirectory, "schtasks.exe");

        public Schtasks(HostLog log)
        {
            _log = log;
        }

        /// <summary>Creates or replaces the task from an XML file (/F).</summary>
        public SchtasksResult Create(string taskName, string xmlPath, TimeSpan limit) =>
            Invoke("/Create /TN \"" + taskName + "\" /XML \"" + xmlPath + "\" /F", limit);

        /// <summary>Starts the task once.</summary>
        public SchtasksResult RunNow(string taskName, TimeSpan limit) => Invoke("/Run /TN \"" + taskName + "\"", limit);

        /// <summary>Deletes the task (/F, no prompt).</summary>
        public SchtasksResult Delete(string taskName, TimeSpan limit) => Invoke("/Delete /TN \"" + taskName + "\" /F", limit);

        /// <summary>True when the task is registered; false when it is absent, or when the query did not finish in time.</summary>
        public bool Exists(string taskName, TimeSpan limit) => Invoke("/Query /TN \"" + taskName + "\"", limit).Succeeded;

        private SchtasksResult Invoke(string arguments, TimeSpan limit)
        {
            var info = new ProcessStartInfo(_executable, arguments)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };

            using (var process = Process.Start(info))
            {
                var output = process.StandardOutput.ReadToEndAsync();
                var error = process.StandardError.ReadToEndAsync();
                if (!process.WaitForExit(Math.Max(0, (int)Math.Min(int.MaxValue, limit.TotalMilliseconds))))
                {
                    TryKill(process);
                    _log.Write("schtasks " + arguments + " -> timed out after " + (int)limit.TotalSeconds + " s; killed.");
                    return new SchtasksResult(-1, "timed out");
                }

                Task.WaitAll(new Task[] { output, error }, 5000);
                var text = JoinLines(Completed(output) + "\n" + Completed(error));
                _log.Write("schtasks " + arguments + " -> exit " + process.ExitCode + (text.Length > 0 ? ": " + text : string.Empty));
                return new SchtasksResult(process.ExitCode, text);
            }
        }

        /// <summary>Joins the non-empty lines of the output with " | " so that one run is one log line.</summary>
        private static string JoinLines(string text) =>
            string.Join(
                " | ",
                text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(line => line.Trim())
                    .Where(line => line.Length > 0));

        private static string Completed(Task<string> task) =>
            task.Status == TaskStatus.RanToCompletion ? task.Result : string.Empty;

        private static void TryKill(Process process)
        {
            try
            {
                process.Kill();
            }
            catch (Exception)
            {
                // Already exited.
            }
        }
    }
}
