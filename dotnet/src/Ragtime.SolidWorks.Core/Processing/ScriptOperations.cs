using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    internal sealed class ScriptOperations
    {
        private readonly ISolidWorksInstanceManager _instances;
        private readonly DocumentRegistry _documents;
        private readonly IScriptRunner _runner;
        private readonly string _workFolder;
        private readonly Action<string>? _log;

        public ScriptOperations(ISolidWorksInstanceManager instances, DocumentRegistry documents, IScriptRunner runner, string workFolder, Action<string>? log)
        {
            _instances = instances;
            _documents = documents;
            _runner = runner;
            _workFolder = workFolder;
            _log = log;
        }

        public async Task<InstanceRun<JObject>> ExecuteAsync(InstanceTarget target, ParsedRequest request, Deadline deadline, CancellationToken token)
        {
            var source = ArgumentReader.RequiredString(request.Arguments, "source");
            var purpose = ArgumentReader.RequiredString(request.Arguments, "purpose");
            var arguments = (JObject?)request.Arguments["arguments"] ?? new JObject();
            var runId = Guid.NewGuid().ToString("N");
            var watch = Stopwatch.StartNew();
            var count = 0;
            ScriptRunOutcome? outcome = null;
            try
            {
                var captured = await _instances.RunAsync(target, request.Context, deadline, session =>
                {
                    var before = session.ListOpenDocuments();
                    try
                    {
                        return (Before: before, State: session.CaptureScriptState());
                    }
                    catch (Exception)
                    {
                        throw HostError.Coded(ErrorCodes.SolidworksUnavailable, "Could not capture SolidWorks application state.");
                    }
                }, token).ConfigureAwait(false);

                var selected = InstanceTarget.Specific(captured.Instance.InstanceId);
                InstanceRun<ScriptRunOutcome> run;
                string? recovery = null;
                HostErrorException? timeoutError = null;
                var restoreFailed = false;
                try
                {
                    run = await _instances.RunAsync(selected, request.Context, deadline, session =>
                    {
                        var timeout = deadline.Remaining - TimeSpan.FromSeconds(5);
                        if (timeout <= TimeSpan.FromSeconds(5))
                            throw HostError.Coded(ErrorCodes.OperationTimeout, "Not enough time remains to run the script.",
                                new JObject { ["instance"] = ScriptRecovery.ProbeOk });
                        var script = new ScriptRunRequest(runId, source, purpose, arguments, session.ProcessId,
                            session.ProcessStartUtc, session.IsAttached, _workFolder, timeout);
                        return _runner.Run(script, token);
                    }, token).ConfigureAwait(false);
                    outcome = run.Value;
                }
                catch (HostErrorException error) when (error.Code == ErrorCodes.OperationTimeout)
                {
                    timeoutError = error;
                    throw;
                }
                finally
                {
                    try
                    {
                        await _instances.RunAsync(selected, request.Context, new Deadline(10, new SystemMonotonicClock()), session =>
                        {
                            try
                            {
                                session.RestoreScriptState(captured.Value.State);
                            }
                            finally
                            {
                                if (captured.Instance.Owned)
                                {
                                    try { session.ReapplyWindowHiding(); }
                                    catch (Exception) { /* Window hiding is best effort. */ }
                                }
                            }
                            return true;
                        }, CancellationToken.None).ConfigureAwait(false);
                    }
                    catch (Exception)
                    {
                        restoreFailed = true;
                    }
                    if (restoreFailed || outcome?.Status == ScriptRunStatus.TimedOut)
                        recovery = await _instances.RecoverAfterScriptAsync(captured.Instance.InstanceId, CancellationToken.None).ConfigureAwait(false);
                    if (timeoutError != null)
                        timeoutError.Details["instance"] = recovery ?? ScriptRecovery.ProbeOk;
                    if (restoreFailed && outcome?.Status != ScriptRunStatus.TimedOut && timeoutError == null)
                    {
                        try { _log?.Invoke($"script.run run={runId} restore_failed instance={recovery}"); }
                        catch (Exception) { /* Logging must not affect the response. */ }
                    }
                }

                if (recovery == null || recovery == ScriptRecovery.ProbeOk)
                {
                    try
                    {
                        var listed = await _instances.RunAsync(selected, request.Context, deadline,
                            session => session.ListOpenDocuments(), token).ConfigureAwait(false);
                        _documents.RetainInstances(_instances.GetInstances().Select(instance => instance.InstanceId));
                        _documents.Synchronize(run.Instance.InstanceId, captured.Value.Before);
                        var beforeIds = new HashSet<string>(_documents.Snapshot()
                            .Where(entry => entry.InstanceId == run.Instance.InstanceId).Select(entry => entry.DocumentId));
                        _documents.Synchronize(run.Instance.InstanceId, listed.Value);
                        foreach (var info in listed.Value)
                        {
                            var entry = _documents.Register(run.Instance.InstanceId, info, false);
                            if (run.Instance.Owned && !beforeIds.Contains(entry.DocumentId))
                                _documents.Register(run.Instance.InstanceId, info, true);
                        }
                    }
                    catch (HostErrorException) when (outcome.Status == ScriptRunStatus.TimedOut) { }
                }

                var docs = _documents.Snapshot().Where(entry => entry.InstanceId == run.Instance.InstanceId)
                    .Select(entry => entry.ToItem()).ToList();
                count = docs.Count;
                if (outcome.Status == ScriptRunStatus.TimedOut)
                    throw HostError.Coded(ErrorCodes.OperationTimeout, "The script timed out.",
                        new JObject { ["instance"] = recovery, ["log"] = JArray.FromObject(outcome.Log) });
                if (outcome.Status != ScriptRunStatus.Completed)
                    throw MapError(outcome);

                if (restoreFailed && (recovery == ScriptRecovery.Restarted || recovery == ScriptRecovery.Stuck))
                {
                    var message = recovery == ScriptRecovery.Restarted
                        ? "The script finished, but SolidWorks did not respond afterwards; the instance was restarted (unsaved documents in it are lost)."
                        : "The script finished, but SolidWorks did not respond afterwards; the instance is marked stuck.";
                    throw HostError.Coded(ErrorCodes.SolidworksUnavailable, message,
                        new JObject { ["instance"] = recovery, ["log"] = JArray.FromObject(outcome.Log), ["message"] = message });
                }

                return new InstanceRun<JObject>(run.Instance, new JObject
                {
                    ["result"] = outcome.Result ?? JValue.CreateNull(),
                    ["log"] = JArray.FromObject(outcome.Log),
                    ["log_truncated"] = outcome.LogTruncated,
                    ["documents"] = WireJson.ToToken(docs),
                    ["instance_id"] = run.Instance.InstanceId,
                    ["compile_ms"] = outcome.CompileMilliseconds,
                    ["run_ms"] = outcome.RunMilliseconds,
                });
            }
            finally
            {
                if (_log != null)
                {
                    using (var sha = SHA256.Create())
                    {
                        var hash = BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(source))).Replace("-", "").ToLowerInvariant();
                        var summaryOutcome = outcome?.Status.ToString() ?? "NoOutcome";
                        try { _log($"script.run run={runId} sha256={hash} purpose={JsonConvert.ToString(purpose)} outcome={summaryOutcome} elapsed_ms={watch.ElapsedMilliseconds} documents={count}"); }
                        catch (Exception) { /* Logging must not affect the response. */ }
                    }
                }
            }
        }

        private static HostErrorException MapError(ScriptRunOutcome outcome)
        {
            switch (outcome.Status)
            {
                case ScriptRunStatus.CompileFailed:
                    var errors = new JArray(outcome.CompileErrors.Select(error =>
                    {
                        var item = new JObject { ["line"] = error.Line, ["column"] = error.Column, ["code"] = error.Code, ["message"] = error.Message };
                        if (error.Hint != null) item["hint"] = error.Hint;
                        return item;
                    }));
                    return HostError.Coded(ErrorCodes.ScriptCompileFailed, "Script compilation failed.", new JObject { ["errors"] = errors });
                case ScriptRunStatus.CompileUnavailable:
                    return HostError.Coded(ErrorCodes.ScriptCompileUnavailable, outcome.Message ?? "Script compilation is unavailable.");
                case ScriptRunStatus.Invalid:
                    return HostError.Coded(ErrorCodes.ScriptInvalid, outcome.Message ?? "The script is invalid.");
                case ScriptRunStatus.AttachFailed:
                    return HostError.Coded(ErrorCodes.SolidworksUnavailable, outcome.Message ?? "Could not attach to SolidWorks.");
                case ScriptRunStatus.Failed:
                    var details = new JObject { ["log"] = JArray.FromObject(outcome.Log) };
                    if (outcome.ExceptionType != null) details["exception_type"] = outcome.ExceptionType;
                    if (outcome.Message != null) details["message"] = outcome.Message;
                    if (outcome.Stack != null) details["stack"] = outcome.Stack;
                    if (outcome.ExitCode != null) details["exit_code"] = outcome.ExitCode.Value;
                    return HostError.Coded(ErrorCodes.ScriptFailed, outcome.Message ?? "The script failed.", details);
                default:
                    throw new InvalidOperationException("Unknown script outcome.");
            }
        }
    }
}
