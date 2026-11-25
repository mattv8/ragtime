using System;
using System.Diagnostics;
using System.IO.Pipes;
using System.Linq;
using System.Threading;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// <c>stop</c>: asks a running host to exit through host.shutdown, then waits up to 30 s for the single-instance
    /// mutex to clear. No running host is success. <c>stop --if-idle N</c> asks the host to exit only when it has been
    /// idle for N seconds, and never stops without a confirmed reply.
    /// </summary>
    internal static class StopCommand
    {
        private static readonly TimeSpan StopWait = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan IfIdleExitWait = TimeSpan.FromSeconds(90);
        private static readonly TimeSpan ShutdownReplyTimeout = TimeSpan.FromSeconds(10);
        private const int PollMilliseconds = 500;

        public static int Run(HostEnvironment env)
        {
            var failure = Stop(env);
            if (failure != null)
            {
                return ClientOutcome.Print(failure, ExitCodes.Failure);
            }

            return ClientOutcome.Print(new JObject { ["stopped"] = WireJson.Value(true) }, ExitCodes.Success);
        }

        /// <summary>Stops the host. Returns null on success, or the error envelope to print.</summary>
        public static JObject? Stop(HostEnvironment env)
        {
            var watch = Stopwatch.StartNew();
            RequestShutdown(env);

            if (!WaitForMutexClear(env, watch, StopWait))
            {
                env.Log.Write("stop: the host still holds its mutex after 30 s.");
                return ClientOutcome.Failure(
                    env,
                    null,
                    null,
                    ErrorCodes.OperationTimeout,
                    "The SolidWorks agent did not stop within 30 seconds.",
                    watch.ElapsedMilliseconds);
            }

            env.Log.Write("stop: no host is running for this user.");
            return null;
        }

        /// <summary>
        /// <c>stop --if-idle N</c>: exit codes 0 (stopped, or not running), 3 (the host is busy and still running) and
        /// 2 (not confirmed idle or stopped). Prints one JSON line.
        /// </summary>
        public static int RunIfIdle(HostEnvironment env, int ifIdleSeconds)
        {
            var watch = Stopwatch.StartNew();
            NamedPipeClientStream? pipe;
            try
            {
                pipe = PipeClient.TryConnect(env.PipeName, env.Sid, PipeClient.ProbeTimeout);
            }
            catch (PipeOwnerMismatchException)
            {
                env.Log.Write("stop --if-idle: the pipe is owned by another account; nothing was stopped.");
                return ClientOutcome.Print(ClientOutcome.OwnerMismatch(env, null, null, watch.ElapsedMilliseconds), ExitCodes.Failure);
            }

            if (pipe == null)
            {
                if (!HostMutex.IsHeld(env.MutexName))
                {
                    env.Log.Write("stop --if-idle: no host is running for this user.");
                    return ClientOutcome.Print(new JObject { ["stopped"] = WireJson.Value(true) }, ExitCodes.Success);
                }

                env.Log.Write("stop --if-idle: the host holds its mutex but does not answer on the pipe; nothing was stopped.");
                return ClientOutcome.Print(
                    ClientOutcome.Failure(
                        env,
                        null,
                        null,
                        ErrorCodes.HostNotRunning,
                        "The SolidWorks agent is running but does not answer. Nothing was stopped.",
                        watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            var id = Guid.NewGuid().ToString();
            StopReply reply;
            using (pipe)
            {
                reply = ExchangeIfIdle(env, pipe, id, ifIdleSeconds);
            }

            switch (reply.Kind)
            {
                case StopReplyKind.Busy:
                    env.Log.Write("stop --if-idle: the host is busy; it was left running.");
                    return ClientOutcome.Print(
                        new JObject
                        {
                            ["stopped"] = WireJson.Value(false),
                            ["busy"] = WireJson.Value(true),
                            ["reasons"] = new JArray(reply.Reasons),
                        },
                        ExitCodes.Busy);
                case StopReplyKind.Failed:
                    // The code comes from the host reply: log it only when it is a known contract code.
                    var loggedCode = reply.Code != null && ErrorCodes.All.Contains(reply.Code) ? reply.Code : "other";
                    env.Log.Write("stop --if-idle: the host refused or did not confirm the stop: " + loggedCode);
                    return ClientOutcome.Print(
                        ClientOutcome.Failure(env, id, null, reply.Code!, reply.Message!, watch.ElapsedMilliseconds),
                        ExitCodes.Failure);
            }

            if (!WaitForMutexClear(env, Stopwatch.StartNew(), IfIdleExitWait))
            {
                env.Log.Write("stop --if-idle: the host accepted the stop but still holds its mutex after 90 s.");
                return ClientOutcome.Print(
                    ClientOutcome.Failure(
                        env,
                        id,
                        null,
                        ErrorCodes.OperationTimeout,
                        "The SolidWorks agent did not stop within 90 seconds.",
                        watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            env.Log.Write("stop --if-idle: the idle host stopped.");
            return ClientOutcome.Print(new JObject { ["stopped"] = WireJson.Value(true) }, ExitCodes.Success);
        }

        private static void RequestShutdown(HostEnvironment env)
        {
            NamedPipeClientStream? pipe;
            try
            {
                pipe = PipeClient.TryConnect(env.PipeName, env.Sid, PipeClient.ProbeTimeout);
            }
            catch (PipeOwnerMismatchException)
            {
                // Nothing is sent to a pipe that another account owns; the mutex below decides whether a host of this user runs.
                env.Log.Write("stop: the pipe is owned by another account; no shutdown request was sent.");
                return;
            }

            if (pipe == null)
            {
                env.Log.Write("stop: no host answers on the pipe.");
                return;
            }

            using (pipe)
            {
                try
                {
                    var reply = PipeClient.ExchangeAsync(pipe, ClientRequests.ShutdownLine(Guid.NewGuid().ToString()), ShutdownReplyTimeout)
                        .GetAwaiter()
                        .GetResult();
                    env.Log.Write(reply == null ? "stop: host.shutdown sent; no reply line." : "stop: host.shutdown answered.");
                }
                catch (Exception exception)
                {
                    env.Log.WriteException(exception);
                }
            }
        }

        /// <summary>Sends the conditional shutdown and reads the reply. Any exception becomes a failed reply.</summary>
        private static StopReply ExchangeIfIdle(HostEnvironment env, NamedPipeClientStream pipe, string id, int ifIdleSeconds)
        {
            string? line;
            try
            {
                line = PipeClient.ExchangeAsync(pipe, ClientRequests.ShutdownLine(id, ifIdleSeconds), ShutdownReplyTimeout)
                    .GetAwaiter()
                    .GetResult();
            }
            catch (Exception exception)
            {
                env.Log.WriteException(exception);
                line = null;
            }

            return StopRequest.Interpret(line, id);
        }

        /// <summary>Polls the single-instance mutex until it clears or <paramref name="limit"/> passes on <paramref name="watch"/>.</summary>
        private static bool WaitForMutexClear(HostEnvironment env, Stopwatch watch, TimeSpan limit)
        {
            while (HostMutex.IsHeld(env.MutexName))
            {
                if (watch.Elapsed >= limit)
                {
                    return false;
                }

                Thread.Sleep(PollMilliseconds);
            }

            return true;
        }
    }
}
