using System;
using System.Collections.Generic;
using System.Threading;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>One <c>script.run</c> request, as Core hands it to the runner. All values are validated by Core.</summary>
    public sealed class ScriptRunRequest
    {
        /// <summary>Creates a request.</summary>
        public ScriptRunRequest(
            string runId,
            string source,
            string purpose,
            JObject arguments,
            int processId,
            DateTime processStartUtc,
            bool attached,
            string workFolder,
            TimeSpan timeout)
        {
            RunId = runId ?? throw new ArgumentNullException(nameof(runId));
            Source = source ?? throw new ArgumentNullException(nameof(source));
            Purpose = purpose ?? throw new ArgumentNullException(nameof(purpose));
            Arguments = arguments ?? throw new ArgumentNullException(nameof(arguments));
            ProcessId = processId;
            ProcessStartUtc = processStartUtc;
            Attached = attached;
            WorkFolder = workFolder ?? throw new ArgumentNullException(nameof(workFolder));
            Timeout = timeout;
        }

        /// <summary>Unique run id: 32 lowercase hex characters. Names the run folder.</summary>
        public string RunId { get; }

        /// <summary>C# 5 source of the script.</summary>
        public string Source { get; }

        /// <summary>The agent's one-line purpose. Logged JSON-encoded; never the source.</summary>
        public string Purpose { get; }

        /// <summary>The script's own arguments (<c>ScriptContext.Arguments</c>).</summary>
        public JObject Arguments { get; }

        /// <summary>SLDWORKS process the worker attaches to (Running Object Table moniker SolidWorks_PID_&lt;pid&gt;).</summary>
        public int ProcessId { get; }

        /// <summary>Start time of that process; the worker refuses a process with another start time (pid reuse).</summary>
        public DateTime ProcessStartUtc { get; }

        /// <summary>True in attach mode (a user's interactive SolidWorks): UserControl/Visible are snapshotted and restored.</summary>
        public bool Attached { get; }

        /// <summary>Host work folder (<c>ScriptContext.WorkFolder</c>).</summary>
        public string WorkFolder { get; }

        /// <summary>Time the runner may spend in total; the worker is killed when it passes.</summary>
        public TimeSpan Timeout { get; }
    }

    /// <summary>How a script run ended.</summary>
    public enum ScriptRunStatus
    {
        /// <summary>Run returned; <see cref="ScriptRunOutcome.Result"/> holds its value.</summary>
        Completed,

        /// <summary>The compiler reported errors (<see cref="ScriptRunOutcome.CompileErrors"/>).</summary>
        CompileFailed,

        /// <summary>The compiler did not run or produced no assembly (for example blocked by security software).</summary>
        CompileUnavailable,

        /// <summary>No <c>public static class RagtimeScript</c> with <c>public static object Run(ScriptContext)</c>.</summary>
        Invalid,

        /// <summary>
        /// Run threw, returned a COM object, or produced a result over the size cap, or the worker exited without a result
        /// (<see cref="ScriptRunOutcome.ExitCode"/>).
        /// </summary>
        Failed,

        /// <summary>The timeout passed; the worker and its children were killed.</summary>
        TimedOut,

        /// <summary>The worker could not attach to the SolidWorks process.</summary>
        AttachFailed,
    }

    /// <summary>One compiler diagnostic, with line and column in the agent's source (1-based).</summary>
    public sealed class ScriptCompileError
    {
        /// <summary>Line in the source, 1-based.</summary>
        public int Line { get; set; }

        /// <summary>Column in the source, 1-based.</summary>
        public int Column { get; set; }

        /// <summary>Compiler code such as CS1056.</summary>
        public string Code { get; set; } = string.Empty;

        /// <summary>Compiler message.</summary>
        public string Message { get; set; } = string.Empty;

        /// <summary>"C# 5 only; see the scripts guidance." for newer-syntax codes; otherwise null.</summary>
        public string? Hint { get; set; }
    }

    /// <summary>Everything a run produced. The runner never throws for script problems; it reports them here.</summary>
    public sealed class ScriptRunOutcome
    {
        /// <summary>How the run ended.</summary>
        public ScriptRunStatus Status { get; set; }

        /// <summary>Run's return value as JSON (Completed only); null for a null return.</summary>
        public JToken? Result { get; set; }

        /// <summary>Lines from <c>ScriptContext.Log</c>, already capped at 2,000 lines / 256 KB by the worker.</summary>
        public List<string> Log { get; set; } = new List<string>();

        /// <summary>True when the worker dropped log lines at the cap.</summary>
        public bool LogTruncated { get; set; }

        /// <summary>Compiler diagnostics (CompileFailed only).</summary>
        public List<ScriptCompileError> CompileErrors { get; set; } = new List<ScriptCompileError>();

        /// <summary>Exception type name (Failed only, when Run threw).</summary>
        public string? ExceptionType { get; set; }

        /// <summary>Human-readable failure message (Failed, Invalid, AttachFailed, CompileUnavailable).</summary>
        public string? Message { get; set; }

        /// <summary>Stack trace trimmed to the script's frames (Failed only, when Run threw).</summary>
        public string? Stack { get; set; }

        /// <summary>Worker exit code when it exited without writing a result.</summary>
        public int? ExitCode { get; set; }

        /// <summary>Compile time in milliseconds.</summary>
        public long CompileMilliseconds { get; set; }

        /// <summary>Run time in milliseconds (from the call of Run to its return).</summary>
        public long RunMilliseconds { get; set; }
    }

    /// <summary>
    /// Runs one script in a separate worker process attached to the given SolidWorks process. Synchronous: Core calls it
    /// inside the instance's work item, so no other call reaches that instance while the script runs. It must return
    /// within <see cref="ScriptRunRequest.Timeout"/> plus a few seconds: on timeout it kills the worker (and its children)
    /// and returns <see cref="ScriptRunStatus.TimedOut"/> with the log so far. Implemented by the host.
    /// </summary>
    public interface IScriptRunner
    {
        /// <summary>Runs the script. Throws only for host faults (for example the run folder cannot be created).</summary>
        ScriptRunOutcome Run(ScriptRunRequest request, CancellationToken cancellationToken);
    }

    /// <summary>What the host did with an instance after a script timed out.</summary>
    public static class ScriptRecovery
    {
        /// <summary>The instance answered a short probe; nothing else was done.</summary>
        public const string ProbeOk = "probe_ok";

        /// <summary>An owned instance did not answer; its process was killed and the record retired.</summary>
        public const string Restarted = "restarted";

        /// <summary>An attached instance did not answer; it is marked stuck and was not touched.</summary>
        public const string Stuck = "stuck";
    }
}
