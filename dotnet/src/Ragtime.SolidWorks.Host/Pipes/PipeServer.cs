using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Security.AccessControl;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>The pipe name is served by another process, so this host cannot listen on it.</summary>
    internal sealed class PipeNameInUseException : Exception
    {
        public PipeNameInUseException(string message)
            : base(message)
        {
        }
    }

    /// <summary>
    /// Accepts pipe connections for the current user and serves one request per connection. A new server instance
    /// is created as soon as one connects, so the name stays served and several connections run concurrently. At most
    /// <see cref="MaxConcurrentConnections"/> connections are served at once; others get a busy envelope. A failing
    /// connection never stops the accept loop.
    /// </summary>
    internal sealed class PipeServer : IDisposable
    {
        /// <summary>Connections served at the same time; further connections are refused with a busy envelope.</summary>
        public const int MaxConcurrentConnections = 32;

        private const int BufferBytes = 64 * 1024;
        private const int ClipLength = 64;
        private static readonly TimeSpan RequestTimeout = TimeSpan.FromSeconds(10);
        private static readonly TimeSpan ResponseTimeout = TimeSpan.FromSeconds(10);
        private static readonly TimeSpan RetryDelay = TimeSpan.FromMilliseconds(250);

        private readonly string _pipeName;
        private readonly PipeSecurity _security;
        private readonly RequestProcessor _processor;
        private readonly string _hostVersion;
        private readonly HostLog _log;
        private readonly CancellationTokenSource _serviceCancellation = new CancellationTokenSource();
        private readonly HashSet<Task> _inFlight = new HashSet<Task>();
        private readonly object _gate = new object();
        private NamedPipeServerStream? _pending;
        private NamedPipeServerStream? _waiting;
        private Task? _acceptLoop;
        private bool _accepting;
        private int _active;

        /// <summary>Creates the server. Call <see cref="Start"/> to begin accepting.</summary>
        public PipeServer(string pipeName, PipeSecurity security, RequestProcessor processor, string hostVersion, HostLog log)
        {
            _pipeName = pipeName;
            _security = security;
            _processor = processor;
            _hostVersion = hostVersion;
            _log = log;
        }

        /// <summary>
        /// Creates the first listener (fails with <see cref="PipeNameInUseException"/> when the name is taken) and starts
        /// the accept loop on a worker thread.
        /// </summary>
        public void Start()
        {
            _pending = FirstPipeInstance.Create(_pipeName, _security, BufferBytes);
            lock (_gate)
            {
                _accepting = true;
            }

            _log.Write("pipe: listening on " + _pipeName);
            _acceptLoop = Task.Run(() => AcceptLoopAsync());
        }

        /// <summary>Stops creating server instances and cancels the wait for the next client.</summary>
        public void StopAccepting()
        {
            NamedPipeServerStream? waiting;
            lock (_gate)
            {
                _accepting = false;
                waiting = _waiting;
                _waiting = null;
            }

            if (waiting != null)
            {
                TimedIo.CloseQuietly(waiting);
            }
        }

        /// <summary>Waits up to <paramref name="timeout"/> for connections in progress, then cancels their work.</summary>
        public async Task DrainAsync(TimeSpan timeout)
        {
            Task[] pending;
            lock (_gate)
            {
                pending = new List<Task>(_inFlight).ToArray();
            }

            await Task.WhenAny(Task.WhenAll(pending), Task.Delay(timeout)).ConfigureAwait(false);
            _serviceCancellation.Cancel();
        }

        /// <summary>Stops accepting and waits (bounded) for the accept loop, which closes every listener instance it holds.</summary>
        public void Dispose()
        {
            StopAccepting();
            var loop = _acceptLoop;
            if (loop != null)
            {
                _ = loop.Wait(ResponseTimeout);
            }
        }

        private async Task AcceptLoopAsync()
        {
            try
            {
                while (IsAccepting())
                {
                    var client = await AcceptOneAsync().ConfigureAwait(false);
                    if (client == null)
                    {
                        continue;
                    }

                    if (TryReserveSlot())
                    {
                        Track(HandleConnectionAsync(client));
                    }
                    else
                    {
                        _log.Write("pipe: connection refused; the host already serves " + MaxConcurrentConnections + " connections.");
                        _ = Task.Run(() => RejectBusyAsync(client));
                    }
                }
            }
            finally
            {
                var leftover = TakePending();
                if (leftover != null)
                {
                    TimedIo.CloseQuietly(leftover);
                }
            }
        }

        private async Task<NamedPipeServerStream?> AcceptOneAsync()
        {
            NamedPipeServerStream? instance = null;
            try
            {
                instance = TakePending() ?? CreateInstance();
                if (!Park(instance))
                {
                    TimedIo.CloseQuietly(instance);
                    return null;
                }

                await WaitForClientAsync(instance).ConfigureAwait(false);
                Unpark();

                // Invariant: the pipe name always has an open server handle. The replacement listener is created before a
                // connected instance is handed to its handler; while it cannot be created, the connected instance stays open.
                var replacement = await CreateReplacementAsync().ConfigureAwait(false);
                if (replacement == null)
                {
                    TimedIo.CloseQuietly(instance);
                    return null;
                }

                _pending = replacement;
                return instance;
            }
            catch (Exception exception)
            {
                Unpark();
                if (IsAccepting())
                {
                    _log.WriteException(exception);
                    await Task.Delay(RetryDelay).ConfigureAwait(false);
                }

                if (instance != null)
                {
                    TimedIo.CloseQuietly(instance);
                }

                return null;
            }
        }

        /// <summary>
        /// Creates the replacement listener, retrying while the server accepts. Returns null when the server stopped
        /// accepting first. Failures are logged and retried; the connected instance is kept open meanwhile.
        /// </summary>
        private async Task<NamedPipeServerStream?> CreateReplacementAsync()
        {
            while (IsAccepting())
            {
                try
                {
                    return CreateInstance();
                }
                catch (Exception exception)
                {
                    _log.WriteException(exception);
                    await Task.Delay(RetryDelay).ConfigureAwait(false);
                }
            }

            return null;
        }

        /// <summary>Creates the next listener with the user-only security descriptor; the first one is made by <see cref="FirstPipeInstance"/>.</summary>
        private NamedPipeServerStream CreateInstance() =>
            new NamedPipeServerStream(
                _pipeName,
                PipeDirection.InOut,
                NamedPipeServerStream.MaxAllowedServerInstances,
                PipeTransmissionMode.Byte,
                PipeOptions.Asynchronous,
                BufferBytes,
                BufferBytes,
                _security);

        private static Task WaitForClientAsync(NamedPipeServerStream instance)
        {
            var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            instance.BeginWaitForConnection(
                result =>
                {
                    try
                    {
                        instance.EndWaitForConnection(result);
                        completion.TrySetResult(true);
                    }
                    catch (Exception exception)
                    {
                        completion.TrySetException(exception);
                    }
                },
                null);
            return completion.Task;
        }

        private async Task HandleConnectionAsync(NamedPipeServerStream client)
        {
            var watch = Stopwatch.StartNew();
            try
            {
                var response = await ReadAndAnswerAsync(client, watch).ConfigureAwait(false);
                if (response != null)
                {
                    await SendResponseAsync(client, response).ConfigureAwait(false);
                }
            }
            finally
            {
                Close(client);
                ReleaseSlot();
            }
        }

        private async Task<string?> ReadAndAnswerAsync(NamedPipeServerStream client, Stopwatch watch)
        {
            try
            {
                var read = await TimedIo.RunAsync(client, RequestTimeout, () => LineReader.ReadAsync(client, ProtocolInfo.MaxRequestBytes))
                    .ConfigureAwait(false);
                return await AnswerAsync(read, watch).ConfigureAwait(false);
            }
            catch (TimeoutException)
            {
                _log.Write("request: the client sent no complete line within 10 s; connection closed.");
                return null;
            }
            catch (Exception exception) when (IsClientGone(exception))
            {
                _log.Write("client disconnected before the request was complete.");
                return null;
            }
            catch (Exception exception)
            {
                _log.WriteException(exception);
                return null;
            }
        }

        private async Task SendResponseAsync(NamedPipeServerStream client, string response)
        {
            try
            {
                await SendAsync(client, response).ConfigureAwait(false);
            }
            catch (TimeoutException)
            {
                _log.Write("response: the client did not accept the response within 10 s; connection closed.");
            }
            catch (Exception exception) when (IsClientGone(exception))
            {
                _log.Write("client disconnected before the response was written: " + DescribeEnvelope(response));
            }
            catch (Exception exception)
            {
                _log.WriteException(exception);
            }
        }

        private async Task<string?> AnswerAsync(LineReadResult read, Stopwatch watch)
        {
            if (read.Kind == LineReadKind.Closed)
            {
                _log.Write("request: the client closed the pipe without a request line.");
                return null;
            }

            if (read.Kind == LineReadKind.TooLong)
            {
                _log.Write("request rejected: larger than 64 KiB; the processor was not called.");
                return InvalidRequestLine("The request is larger than 64 KiB.", watch.ElapsedMilliseconds);
            }

            if (!Utf8Text.TryDecode(read.Bytes, out var line))
            {
                _log.Write("request rejected: not valid UTF-8; the processor was not called.");
                return InvalidRequestLine("The request is not valid UTF-8.", watch.ElapsedMilliseconds);
            }

            var response = await _processor.ProcessLineAsync(line, _serviceCancellation.Token).ConfigureAwait(false);
            LogRequest(line, response, watch.ElapsedMilliseconds);
            return response;
        }

        private string InvalidRequestLine(string message, long elapsedMs) =>
            ResponseEnvelope.ToLine(ClientEnvelopes.Failure(null, null, _hostVersion, ErrorCodes.InvalidRequest, message, elapsedMs));

        private static async Task SendAsync(NamedPipeServerStream client, string response)
        {
            var bytes = Utf8Text.Encode(response + "\n");
            await TimedIo.RunAsync(client, ResponseTimeout, async () =>
            {
                await client.WriteAsync(bytes, 0, bytes.Length).ConfigureAwait(false);
                await client.FlushAsync().ConfigureAwait(false);
                return true;
            }).ConfigureAwait(false);

            // WaitForPipeDrain blocks until the client has read everything; bound it so a stuck client cannot pin the instance.
            var drained = Task.Run(() => client.WaitForPipeDrain());
            await Task.WhenAny(drained, Task.Delay(ResponseTimeout)).ConfigureAwait(false);
            _ = drained.ContinueWith(ObserveFault, TaskContinuationOptions.OnlyOnFaulted);
        }

        private async Task RejectBusyAsync(NamedPipeServerStream client)
        {
            try
            {
                var line = ResponseEnvelope.ToLine(ClientEnvelopes.Failure(null, null, _hostVersion, ErrorCodes.InternalError, "The host is busy.", 0));
                await SendAsync(client, line).ConfigureAwait(false);
            }
            catch (Exception)
            {
                // The client went away or stopped reading; nothing else to report.
            }
            finally
            {
                Close(client);
            }
        }

        private static void ObserveFault(Task faulted)
        {
            _ = faulted.Exception;
        }

        private static bool IsClientGone(Exception exception) =>
            exception is IOException || exception is ObjectDisposedException;

        private void LogRequest(string line, string response, long elapsedMs)
        {
            var operation = Clip(RequestParser.TryReadString(line, "operation"));
            var id = Clip(RequestParser.TryReadString(line, "id"));
            var ok = false;
            var code = "-";
            try
            {
                var envelope = JObject.Parse(response);
                ok = (bool?)envelope["ok"] == true;
                if (!ok)
                {
                    code = Clip((string?)envelope["error"]?["code"]);
                }
            }
            catch (JsonException)
            {
                code = "unparsed";
            }

            _log.Write($"request operation={operation} id={id} ok={ok} code={code} elapsed_ms={elapsedMs}");
        }

        private static string DescribeEnvelope(string response)
        {
            try
            {
                var envelope = JObject.Parse(response);
                return "operation=" + Clip((string?)envelope["operation"]) + " id=" + Clip((string?)envelope["id"]);
            }
            catch (JsonException)
            {
                return "operation=- id=-";
            }
        }

        private static string Clip(string? value)
        {
            if (value == null)
            {
                return "-";
            }

            return value.Length <= ClipLength ? value : value.Substring(0, ClipLength);
        }

        private static void Close(NamedPipeServerStream client)
        {
            try
            {
                client.Disconnect();
            }
            catch (Exception)
            {
                // Not connected any more.
            }

            TimedIo.CloseQuietly(client);
        }

        private void Track(Task task)
        {
            lock (_gate)
            {
                _inFlight.Add(task);
            }

            task.ContinueWith(finished =>
            {
                lock (_gate)
                {
                    _inFlight.Remove(finished);
                }
            }, TaskScheduler.Default);
        }

        private bool TryReserveSlot()
        {
            lock (_gate)
            {
                if (_active >= MaxConcurrentConnections)
                {
                    return false;
                }

                _active++;
                return true;
            }
        }

        private void ReleaseSlot()
        {
            lock (_gate)
            {
                _active--;
            }
        }

        private bool IsAccepting()
        {
            lock (_gate)
            {
                return _accepting;
            }
        }

        private bool Park(NamedPipeServerStream instance)
        {
            lock (_gate)
            {
                if (!_accepting)
                {
                    return false;
                }

                _waiting = instance;
                return true;
            }
        }

        private void Unpark()
        {
            lock (_gate)
            {
                _waiting = null;
            }
        }

        private NamedPipeServerStream? TakePending()
        {
            var pending = _pending;
            _pending = null;
            return pending;
        }
    }
}
