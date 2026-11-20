using System;
using System.Diagnostics;
using System.IO.Pipes;
using System.Threading;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// <c>stop</c>: asks a running host to exit through host.shutdown, then waits up to 30 s for the single-instance
    /// mutex to clear. No running host is success.
    /// </summary>
    internal static class StopCommand
    {
        private static readonly TimeSpan StopWait = TimeSpan.FromSeconds(30);
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

            while (HostMutex.IsHeld(env.MutexName))
            {
                if (watch.Elapsed >= StopWait)
                {
                    env.Log.Write("stop: the host still holds its mutex after 30 s.");
                    return ClientOutcome.Failure(
                        env,
                        null,
                        null,
                        ErrorCodes.OperationTimeout,
                        "The SolidWorks host did not stop within 30 seconds.",
                        watch.ElapsedMilliseconds);
                }

                Thread.Sleep(PollMilliseconds);
            }

            env.Log.Write("stop: no host is running for this user.");
            return null;
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
    }
}
