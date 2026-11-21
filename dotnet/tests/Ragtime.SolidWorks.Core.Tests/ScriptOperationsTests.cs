using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptOperationsTests
    {
        private sealed class Runner : IScriptRunner
        {
            public ScriptRunOutcome Outcome { get; set; } = new ScriptRunOutcome { Status = ScriptRunStatus.Completed };
            public ScriptRunRequest? Request { get; private set; }
            public Action? OnRun { get; set; }
            public ScriptRunOutcome Run(ScriptRunRequest request, CancellationToken cancellationToken)
            {
                Request = request;
                OnRun?.Invoke();
                return Outcome;
            }
        }

        private static async Task<JObject> Call(FakeInstanceManager manager, Runner? runner, List<string>? log = null, bool attach = false, CancellationToken token = default)
        {
            var clock = new FakeClock();
            manager.Clock = clock;
            var processor = new RequestProcessor(manager, new FakeHostFacts(), OperationRegistry.Default, TestData.HostVersion,
                _ => Task.CompletedTask, clock, pdmWorkFolder: "work", scriptRunner: runner,
                hostLog: log == null ? null : new Action<string>(log.Add));
            var context = TestData.Context();
            context["allow_scripts"] = true;
            if (attach) context["instance_mode"] = "attach";
            var args = new JObject { ["source"] = "secret source", ["purpose"] = "A \"purpose\"", ["arguments"] = new JObject { ["a"] = 1 } };
            return JObject.Parse(await processor.ProcessLineAsync(TestData.Request("script.run", args, context), token));
        }

        [Theory]
        [InlineData(ScriptRunStatus.CompileFailed, "script_compile_failed")]
        [InlineData(ScriptRunStatus.CompileUnavailable, "script_compile_unavailable")]
        [InlineData(ScriptRunStatus.Invalid, "script_invalid")]
        [InlineData(ScriptRunStatus.Failed, "script_failed")]
        [InlineData(ScriptRunStatus.AttachFailed, "solidworks_unavailable")]
        public async Task MapsErrors(ScriptRunStatus status, string code)
        {
            var manager = new FakeInstanceManager();
            manager.AddBackground("i1");
            var runner = new Runner { Outcome = new ScriptRunOutcome { Status = status, Message = "message", ExceptionType = "Exception", Stack = "stack", ExitCode = 4 } };
            runner.Outcome.CompileErrors.Add(new ScriptCompileError { Line = 2, Column = 3, Code = "CS1", Message = "bad", Hint = "hint" });
            runner.Outcome.Log.Add("line");
            var response = await Call(manager, runner);
            Assert.Equal(code, (string?)response["error"]!["code"]);
            if (status == ScriptRunStatus.CompileFailed)
                Assert.Equal("hint", (string?)response["error"]!["details"]!["errors"]![0]!["hint"]);
            if (status == ScriptRunStatus.Failed)
                Assert.Equal(4, (int)response["error"]!["details"]!["exit_code"]!);
        }

        [Theory]
        [InlineData(ScriptRecovery.ProbeOk)]
        [InlineData(ScriptRecovery.Restarted)]
        [InlineData(ScriptRecovery.Stuck)]
        public async Task TimeoutReportsRecovery(string recovery)
        {
            var manager = new FakeInstanceManager { Recovery = recovery };
            manager.AddBackground("i1");
            var response = await Call(manager, new Runner { Outcome = new ScriptRunOutcome { Status = ScriptRunStatus.TimedOut } });
            Assert.Equal("operation_timeout", (string?)response["error"]!["code"]);
            Assert.Equal(recovery, (string?)response["error"]!["details"]!["instance"]);
            Assert.Equal(new[] { "i1" }, manager.RecoveredInstances);
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public async Task RunsOnSessionAndReportsDocumentsAndSafeLog(bool attach)
        {
            var manager = new FakeInstanceManager();
            var snapshot = attach ? manager.AddAttached(4242) : manager.AddBackground("i1");
            var session = manager.SessionFor(snapshot.InstanceId);
            var runner = new Runner();
            runner.OnRun = () =>
            {
                Assert.True(manager.RunningWork);
                session.Open.Add(new SessionDocumentInfo("new", @"C:\new.sldprt", "part", null, false, false));
            };
            var log = new List<string>();
            var response = await Call(manager, runner, log, attach);
            Assert.True((bool)response["ok"]!);
            Assert.Equal(session.ProcessId, runner.Request!.ProcessId);
            Assert.Equal(session.ProcessStartUtc, runner.Request.ProcessStartUtc);
            Assert.Equal(attach, runner.Request.Attached);
            Assert.Equal("work", runner.Request.WorkFolder);
            Assert.Equal(TimeSpan.FromSeconds(85), runner.Request.Timeout);
            Assert.Matches("^[0-9a-f]{32}$", runner.Request.RunId);
            Assert.NotEmpty(manager.Targets);
            Assert.Equal(!attach, session.Calls.Contains(nameof(FakeSession.ReapplyWindowHiding)));
            Assert.Equal(!attach, (bool)response["result"]!["documents"]![0]!["opened_by_host"]!);
            Assert.Contains("purpose=" + JsonConvert.ToString("A \"purpose\""), Assert.Single(log));
            Assert.Contains("sha256=f07976861a734242e7cd7f598ff191e0f065b031efe9f1f8e5c50e670a20bdc7", log[0]);
            Assert.DoesNotContain("secret source", log[0]);
        }

        [Fact]
        public async Task MissingRunnerIsInternalError()
        {
            var response = await Call(new FakeInstanceManager(), null);
            Assert.Equal("internal_error", (string?)response["error"]!["code"]);
        }

        [Theory]
        [InlineData(ScriptRunStatus.Completed)]
        [InlineData(ScriptRunStatus.CompileFailed)]
        [InlineData(ScriptRunStatus.CompileUnavailable)]
        [InlineData(ScriptRunStatus.Invalid)]
        [InlineData(ScriptRunStatus.Failed)]
        [InlineData(ScriptRunStatus.AttachFailed)]
        [InlineData(ScriptRunStatus.TimedOut)]
        public async Task RestoresStateAfterEveryOutcome(ScriptRunStatus status)
        {
            var manager = new FakeInstanceManager();
            manager.AddBackground("i1");
            var session = manager.SessionFor("i1");
            session.ScriptState = new ScriptSessionState { DocumentVisiblePart = true, DocumentVisibleAssembly = false, DocumentVisibleDrawing = true };
            var runner = new Runner { Outcome = new ScriptRunOutcome { Status = status } };
            runner.OnRun = () => session.Calls.Add("Run");
            await Call(manager, runner);
            Assert.True(session.Calls.IndexOf("CaptureScriptState") < session.Calls.IndexOf("Run"));
            Assert.True(session.Calls.IndexOf("Run") < session.Calls.IndexOf("RestoreScriptState"));
            Assert.Same(session.ScriptState, session.RestoredScriptState);
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public async Task RestoresStateWhenRunnerThrowsOrCancellationOccurs(bool cancel)
        {
            var manager = new FakeInstanceManager();
            manager.AddBackground("i1");
            var session = manager.SessionFor("i1");
            using (var cancellation = new CancellationTokenSource())
            {
                var runner = new Runner { OnRun = () =>
                {
                    if (cancel) cancellation.Cancel();
                    if (cancel) throw new OperationCanceledException(cancellation.Token);
                    throw new InvalidOperationException("runner failed");
                } };
                await Call(manager, runner, token: cancellation.Token);
            }
            Assert.Contains("RestoreScriptState", session.Calls);
            Assert.Same(session.ScriptState, session.RestoredScriptState);
            Assert.Equal(CancellationToken.None, manager.RunTokens[2]);
            Assert.Equal(TimeSpan.FromSeconds(10), manager.Deadlines[2].Total);
        }

        [Fact]
        public async Task CaptureFailureRefusesRun()
        {
            var manager = new FakeInstanceManager();
            manager.AddBackground("i1");
            var session = manager.SessionFor("i1");
            session.FailScriptStateCapture = true;
            var runner = new Runner();
            var response = await Call(manager, runner);
            Assert.Equal("solidworks_unavailable", (string?)response["error"]!["code"]);
            Assert.Null(runner.Request);
            Assert.DoesNotContain("RestoreScriptState", session.Calls);
        }

        [Fact]
        public async Task RestoresStateWhenRunnerReturnsAfterCancellation()
        {
            var manager = new FakeInstanceManager();
            manager.AddBackground("i1");
            var session = manager.SessionFor("i1");
            using (var cancellation = new CancellationTokenSource())
            {
                var runner = new Runner
                {
                    Outcome = new ScriptRunOutcome { Status = ScriptRunStatus.TimedOut },
                    OnRun = cancellation.Cancel,
                };
                await Call(manager, runner, token: cancellation.Token);
            }
            Assert.Same(session.ScriptState, session.RestoredScriptState);
            Assert.Contains("RestoreScriptState", session.Calls);
        }

        [Fact]
        public async Task CaptureTimeReducesWorkerTimeout()
        {
            var manager = new FakeInstanceManager { CostMs = 10000 };
            manager.AddBackground("i1");
            var runner = new Runner();
            await Call(manager, runner);
            Assert.Equal(TimeSpan.FromSeconds(65), runner.Request!.Timeout);
            Assert.Equal(TimeSpan.FromSeconds(10), manager.Deadlines[2].Total);
        }

        [Theory]
        [InlineData(40000)]
        [InlineData(41000)]
        public async Task ReservePreventsLaunchingTheWorker(int costMs)
        {
            var manager = new FakeInstanceManager { CostMs = costMs };
            manager.AddBackground("i1");
            var runner = new Runner();
            var response = await Call(manager, runner);
            Assert.Null(runner.Request);
            Assert.Equal("operation_timeout", (string?)response["error"]!["code"]);
            Assert.Equal("probe_ok", (string?)response["error"]!["details"]!["instance"]);
            Assert.Same(manager.SessionFor("i1").ScriptState, manager.SessionFor("i1").RestoredScriptState);
        }

        [Theory]
        [InlineData(ScriptRunStatus.Completed, ScriptRecovery.Restarted)]
        [InlineData(ScriptRunStatus.Completed, ScriptRecovery.Stuck)]
        [InlineData(ScriptRunStatus.Completed, ScriptRecovery.ProbeOk)]
        [InlineData(ScriptRunStatus.TimedOut, ScriptRecovery.Restarted)]
        public async Task RestoreTimeoutRecoversExactlyOnce(ScriptRunStatus status, string recovery)
        {
            var manager = new FakeInstanceManager { Recovery = recovery };
            manager.AddBackground("i1");
            manager.SessionFor("i1").FailScriptStateRestore = true;
            var runner = new Runner { Outcome = new ScriptRunOutcome { Status = status } };
            runner.Outcome.Result = new JValue("private result");
            runner.Outcome.Log.Add("script log");
            var log = new List<string>();
            var response = await Call(manager, runner, log);
            Assert.Equal(new[] { "i1" }, manager.RecoveredInstances);
            if (status == ScriptRunStatus.TimedOut)
            {
                Assert.Equal("operation_timeout", (string?)response["error"]!["code"]);
                Assert.Equal("restarted", (string?)response["error"]!["details"]!["instance"]);
            }
            else if (recovery == ScriptRecovery.ProbeOk)
            {
                Assert.True((bool)response["ok"]!);
                Assert.Contains(log, line => line.Contains("restore_failed") && line.Contains("probe_ok"));
            }
            else
            {
                Assert.Equal("solidworks_unavailable", (string?)response["error"]!["code"]);
                var details = response["error"]!["details"]!;
                Assert.Equal(recovery, (string?)details["instance"]);
                Assert.Equal("script log", (string?)details["log"]![0]);
                var message = recovery == ScriptRecovery.Restarted
                    ? "The script finished, but SolidWorks did not respond afterwards; the instance was restarted (unsaved documents in it are lost)."
                    : "The script finished, but SolidWorks did not respond afterwards; the instance is marked stuck.";
                Assert.Equal(message, (string?)details["message"]);
                Assert.DoesNotContain("private result", response.ToString());
            }
        }
    }
}
