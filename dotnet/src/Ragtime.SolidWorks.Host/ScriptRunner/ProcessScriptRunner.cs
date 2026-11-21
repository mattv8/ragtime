using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Runs a script in a <c>script-worker</c> process of this host, attached to the SolidWorks instance. The run folder
    /// holds the inputs, the worker's log and result. The worker starts suspended inside a kill-on-close Job Object, so
    /// nothing it spawns escapes a timeout.
    /// </summary>
    internal sealed class ProcessScriptRunner : IScriptRunner
    {
        private const string WorkerOutputName = "worker.out";
        private static readonly TimeSpan WorkerStopWait = TimeSpan.FromSeconds(5);

        private readonly string _hostExePath;
        private readonly string _scriptsRoot;
        private readonly Action<string> _log;
        private readonly Func<DateTime> _utcNow;

        /// <summary>Creates the runner.</summary>
        /// <param name="hostExePath">This host's executable; the worker is started from it.</param>
        /// <param name="scriptsRoot">%LOCALAPPDATA%\RagtimeSolidWorks\scripts\; run folders are created inside it.</param>
        /// <param name="log">Host log sink.</param>
        /// <param name="utcNow">UTC clock for the worker's deadline.</param>
        public ProcessScriptRunner(string hostExePath, string scriptsRoot, Action<string> log, Func<DateTime> utcNow)
        {
            _hostExePath = hostExePath ?? throw new ArgumentNullException(nameof(hostExePath));
            _scriptsRoot = scriptsRoot ?? throw new ArgumentNullException(nameof(scriptsRoot));
            _log = log ?? throw new ArgumentNullException(nameof(log));
            _utcNow = utcNow ?? throw new ArgumentNullException(nameof(utcNow));
        }

        /// <inheritdoc />
        public ScriptRunOutcome Run(ScriptRunRequest request, CancellationToken cancellationToken)
        {
            RefuseLinkedRoot(request.RunId);
            var folder = CreateRunFolder(request.RunId);
            WriteInputs(folder, request);
            var outcome = Execute(folder, request, cancellationToken);
            if (outcome.Status == ScriptRunStatus.Completed)
            {
                TryDeleteRunFolder(folder);
            }

            return outcome;
        }

        /// <summary>Re-checks the scripts root and its parent before every run; a link refuses the run with a host fault, logged.</summary>
        private void RefuseLinkedRoot(string runId)
        {
            var linked = ScriptRootGuard.FirstLinkedFolder(_scriptsRoot);
            if (linked != null)
            {
                _log("script run " + runId + ": refused; " + linked + " is a junction or link.");
                throw new IOException("Refusing a junction or link as a script folder: " + linked);
            }
        }

        /// <summary>Creates <c>&lt;root&gt;\&lt;runId&gt;\</c> with an ACL that grants FullControl to the current user only.</summary>
        private string CreateRunFolder(string runId)
        {
            if (Path.GetFileName(runId) != runId || runId == "." || runId == "..")
            {
                throw new ArgumentException("The run id is not a valid folder name.", nameof(runId));
            }

            var folder = Path.Combine(_scriptsRoot, runId);
            if (Directory.Exists(folder))
            {
                throw new IOException("The script run folder already exists: " + folder);
            }

            var user = WindowsIdentity.GetCurrent().User ?? throw new InvalidOperationException("The current user has no SID.");
            var security = new DirectorySecurity();
            security.SetOwner(user);
            security.SetAccessRuleProtection(true, false);
            security.AddAccessRule(new FileSystemAccessRule(
                user,
                FileSystemRights.FullControl,
                InheritanceFlags.ContainerInherit | InheritanceFlags.ObjectInherit,
                PropagationFlags.None,
                AccessControlType.Allow));
            Directory.CreateDirectory(folder, security);
            ScriptRootGuard.ThrowIfLinked(folder);
            return folder;
        }

        /// <summary>Writes <c>script.cs</c> (UTF-8, no BOM) and <c>request.json</c>.</summary>
        private void WriteInputs(string folder, ScriptRunRequest request)
        {
            File.WriteAllText(Path.Combine(folder, ScriptWorkerFiles.SourceName), request.Source, new UTF8Encoding(false));
            var deadline = DateTime.SpecifyKind(_utcNow() + request.Timeout, DateTimeKind.Utc);
            ScriptWorkerFiles.WriteRequest(folder, new JObject
            {
                ["run_id"] = request.RunId,
                ["process_id"] = request.ProcessId,
                ["process_start_utc"] = request.ProcessStartUtc.ToUniversalTime().ToString("o", CultureInfo.InvariantCulture),
                ["attached"] = request.Attached,
                ["work_folder"] = request.WorkFolder,
                ["host_folder"] = Path.GetDirectoryName(_hostExePath),
                ["deadline_utc"] = deadline.ToString("o", CultureInfo.InvariantCulture),
                ["arguments"] = (JObject)request.Arguments.DeepClone(),
            });
        }

        private ScriptRunOutcome Execute(string folder, ScriptRunRequest request, CancellationToken cancellationToken)
        {
            using (var job = JobObject.CreateKillOnClose())
            using (var process = StartWorker(request.RunId, folder, job))
            {
                if (!WorkerLauncher.WaitForExit(process, request.Timeout, cancellationToken))
                {
                    job.Dispose();
                    var stopped = WorkerLauncher.WaitForExit(process, WorkerStopWait, CancellationToken.None);
                    _log("script run " + request.RunId + ": timed out after " + request.Timeout.TotalSeconds.ToString("0.#", CultureInfo.InvariantCulture) + " s; "
                        + (stopped ? "the worker and its children were stopped." : "the worker did not exit within 5 s after the stop."));
                    return ScriptOutcomeMapper.TimedOut(ReadLog(folder));
                }

                var exitCode = WorkerLauncher.ExitCode(process);
                job.Dispose();
                return ReadOutcome(folder, exitCode);
            }
        }

        /// <summary>Starts the suspended worker inside the job. Logs and rethrows when it cannot be started.</summary>
        private SafeKernelHandle StartWorker(string runId, string folder, SafeJobHandle job)
        {
            try
            {
                return WorkerLauncher.Start(_hostExePath, folder, Path.Combine(folder, WorkerOutputName), job);
            }
            catch (Exception exception)
            {
                _log("script run " + runId + ": the worker could not be started: " + exception.Message);
                throw;
            }
        }

        private ScriptRunOutcome ReadOutcome(string folder, int exitCode)
        {
            var log = ReadLog(folder);
            if (!File.Exists(Path.Combine(folder, ScriptWorkerFiles.ResultName)))
            {
                return ScriptOutcomeMapper.NoResult(exitCode, log);
            }

            try
            {
                return ScriptOutcomeMapper.FromResult(ScriptWorkerFiles.ReadResult(folder), log);
            }
            catch (Exception exception) when (IsUnreadableResult(exception))
            {
                _log("script run " + Path.GetFileName(folder) + ": result.json is unreadable: " + exception.Message);
                return ScriptOutcomeMapper.Malformed(exitCode, exception.Message, log);
            }
        }

        private static bool IsUnreadableResult(Exception exception) =>
            exception is IOException
            || exception is UnauthorizedAccessException
            || exception is JsonException
            || exception is ScriptResultFormatException;

        /// <summary>Reads the worker's log. A log that cannot be read is logged and treated as empty; it never fails the run.</summary>
        private List<string> ReadLog(string folder)
        {
            try
            {
                return ScriptOutcomeMapper.ReadLog(ScriptWorkerFiles.ReadLogText(folder));
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                _log("script run " + Path.GetFileName(folder) + ": log.jsonl could not be read: " + exception.Message);
                return ScriptOutcomeMapper.ReadLog(null);
            }
        }

        private void TryDeleteRunFolder(string folder)
        {
            try
            {
                Directory.Delete(folder, true);
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                _log("script run " + Path.GetFileName(folder) + ": the run folder could not be deleted: " + exception.Message);
            }
        }
    }
}
