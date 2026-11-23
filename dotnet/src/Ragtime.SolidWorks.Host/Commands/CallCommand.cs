using System;
using System.Diagnostics;
using System.IO.Pipes;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// <c>call</c>: forwards one request to the host and prints the host's response line verbatim (exit 0), or prints
    /// a client-side error envelope (exit 2). Starts the logon task once when the pipe is absent. Every step before the
    /// pipe exchange is bounded by the overall cap (deadline_seconds + 5 s) minus the time already spent.
    /// </summary>
    internal static class CallCommand
    {
        private const string IncompleteResponseMessage = "The host closed the connection before answering completely.";
        private static readonly TimeSpan PipeWait = TimeSpan.FromSeconds(20);

        public static int Run(CommandLine command, HostEnvironment env)
        {
            var watch = Stopwatch.StartNew();
            var log = env.Log;

            string text;
            ParsedRequest parsed;
            JObject root;
            try
            {
                text = ReadRequestText(command.RequestBase64);
                parsed = RequestParser.Parse(text);
                root = JObject.Parse(text);
            }
            catch (HostErrorException error)
            {
                var requestText = TryReadRequestText(command.RequestBase64);
                return ClientOutcome.Print(
                    ResponseEnvelope.FromException(
                        RequestParser.TryReadString(requestText, "id"),
                        RequestParser.TryReadString(requestText, "operation"),
                        error,
                        env.HostVersion,
                        watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            var id = parsed.Id;
            var operation = parsed.Operation;
            var originalDeadline = parsed.Context.DeadlineSeconds;
            var overallCap = CallTiming.OverallCap(originalDeadline);
            log.Write($"call: operation {operation}, deadline {originalDeadline} s");

            if (IsInternal(operation))
            {
                log.Write("call: refused an internal operation; nothing was sent and no host was started.");
                return ClientOutcome.Print(
                    ClientOutcome.Failure(env, id, operation, ErrorCodes.UnknownOperation, "The operation is internal to the host and cannot be called.", watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            try
            {
                return Forward(root, parsed, env, overallCap, watch);
            }
            catch (PipeOwnerMismatchException)
            {
                log.Write("call: the pipe is owned by another account; nothing was sent and no host was started.");
                return ClientOutcome.Print(ClientOutcome.OwnerMismatch(env, id, operation, watch.ElapsedMilliseconds), ExitCodes.Failure);
            }
        }

        private static int Forward(JObject root, ParsedRequest parsed, HostEnvironment env, TimeSpan overallCap, Stopwatch watch)
        {
            var log = env.Log;
            var id = parsed.Id;
            var operation = parsed.Operation;
            var originalDeadline = parsed.Context.DeadlineSeconds;

            var pipe = PipeClient.TryConnect(env.PipeName, env.Sid, Smaller(PipeClient.ProbeTimeout, Left(overallCap, watch)));
            bool? registered = null;
            if (pipe == null)
            {
                if (!HasTime(overallCap, watch))
                {
                    return NotRunning(env, id, operation, registered, watch);
                }

                var schtasks = new Schtasks(log);
                registered = schtasks.Exists(env.TaskName, Left(overallCap, watch));
                if (!HasTime(overallCap, watch))
                {
                    return NotRunning(env, id, operation, registered, watch);
                }

                schtasks.RunNow(env.TaskName, Left(overallCap, watch));
                if (!HasTime(overallCap, watch))
                {
                    return NotRunning(env, id, operation, registered, watch);
                }

                pipe = PipeClient.WaitForPipe(env.PipeName, env.Sid, Smaller(PipeWait, Left(overallCap, watch)));
                if (pipe == null)
                {
                    log.Write("call: the host pipe did not appear.");
                    return NotRunning(env, id, operation, registered, watch);
                }
            }

            using (pipe)
            {
                var forwarded = CallTiming.ForwardedDeadlineSeconds(originalDeadline, watch.Elapsed);
                if (forwarded == null)
                {
                    log.Write("call: no time left for the request after starting the host.");
                    return NotRunning(env, id, operation, registered, watch);
                }

                ((JObject)root["context"]!)["deadline_seconds"] = forwarded.Value;
                return Exchange(pipe, ResponseEnvelope.ToLine(root), env, id, operation, Left(overallCap, watch), watch);
            }
        }

        private static int Exchange(
            NamedPipeClientStream pipe,
            string requestLine,
            HostEnvironment env,
            string id,
            string operation,
            TimeSpan remaining,
            Stopwatch watch)
        {
            string? response;
            try
            {
                response = PipeClient.ExchangeAsync(pipe, requestLine, remaining).GetAwaiter().GetResult();
            }
            catch (TimeoutException)
            {
                env.Log.Write("call: no response before the overall cap of deadline_seconds + 5 s.");
                return ClientOutcome.Print(
                    ClientOutcome.Failure(
                        env,
                        id,
                        operation,
                        ErrorCodes.OperationTimeout,
                        "The SolidWorks host did not answer within the request deadline plus 5 seconds.",
                        watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }
            catch (Exception exception)
            {
                env.Log.WriteException(exception);
                response = null;
            }

            if (response == null || !ClientOutcome.IsCompleteResponse(response, id))
            {
                env.Log.Write("call: the host response was missing, incomplete or for another request.");
                return ClientOutcome.Print(
                    ClientOutcome.Failure(env, id, operation, ErrorCodes.InternalError, IncompleteResponseMessage, watch.ElapsedMilliseconds),
                    ExitCodes.Failure);
            }

            JsonOutput.WriteLine(response);
            return ExitCodes.Success;
        }

        /// <summary>host_not_running for the no-time and no-pipe paths; taskRegistered is null when it was not queried.</summary>
        private static int NotRunning(HostEnvironment env, string id, string operation, bool? registered, Stopwatch watch) =>
            ClientOutcome.Print(ClientOutcome.NotRunning(env, id, operation, registered, watch.ElapsedMilliseconds), ExitCodes.Failure);

        /// <summary>Internal operations are never forwarded: host.* names are reserved for the host command itself.</summary>
        private static bool IsInternal(string operation)
        {
            var definition = OperationRegistry.Default.Find(operation);
            return (definition != null && definition.IsInternal) || operation.StartsWith("host.", StringComparison.Ordinal);
        }

        /// <summary>Best-effort text of the request for echoing its id and operation on errors; null when unreadable.</summary>
        private static string? TryReadRequestText(string? base64)
        {
            try
            {
                return base64 == null ? null : DecodeUtf8(CallRequestArgument.Decode(base64));
            }
            catch (HostErrorException)
            {
                return null;
            }
        }

        private static string ReadRequestText(string? base64)
        {
            if (base64 == null)
            {
                return ReadStdinLine();
            }

            return DecodeUtf8(CallRequestArgument.Decode(base64));
        }

        private static string ReadStdinLine()
        {
            using (var stdin = Console.OpenStandardInput())
            {
                var read = LineReader.ReadAsync(stdin, ProtocolInfo.MaxRequestBytes).GetAwaiter().GetResult();
                if (read.Kind == LineReadKind.TooLong)
                {
                    throw HostError.InvalidRequest("The request is larger than 64 KiB.");
                }

                if (read.Kind == LineReadKind.Closed)
                {
                    throw HostError.InvalidRequest("No request was given. Pass --request-b64 or write the request line to stdin.");
                }

                return DecodeUtf8(read.Bytes);
            }
        }

        private static string DecodeUtf8(byte[] bytes)
        {
            if (!Utf8Text.TryDecode(bytes, out var text))
            {
                throw HostError.InvalidRequest("The request is not valid UTF-8.");
            }

            return text;
        }

        private static TimeSpan Left(TimeSpan cap, Stopwatch watch)
        {
            var left = cap - watch.Elapsed;
            return left > TimeSpan.Zero ? left : TimeSpan.Zero;
        }

        private static bool HasTime(TimeSpan cap, Stopwatch watch) => Left(cap, watch) > TimeSpan.Zero;

        private static TimeSpan Smaller(TimeSpan first, TimeSpan second) => first < second ? first : second;
    }
}
