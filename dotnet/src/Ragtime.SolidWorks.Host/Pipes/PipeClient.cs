using System;
using System.Diagnostics;
using System.IO.Pipes;
using System.Security.Principal;
using System.Threading;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Raised when the pipe answers but its server process does not run as the expected account. Nothing has been sent
    /// to it, and the caller must not start a host because of it.
    /// </summary>
    internal sealed class PipeOwnerMismatchException : Exception
    {
        public PipeOwnerMismatchException()
            : base("The host pipe is owned by another account.")
        {
        }
    }

    /// <summary>Client side of the host pipe: one connection per request, one request line and one response line.</summary>
    internal static class PipeClient
    {
        /// <summary>Connect attempt length for a single probe.</summary>
        public static readonly TimeSpan ProbeTimeout = TimeSpan.FromMilliseconds(500);

        private static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(250);
        private const int MaxResponseBytes = 1024 * 1024;

        /// <summary>
        /// Connects to the pipe with identification impersonation and verifies that the server runs as
        /// <paramref name="expectedSid"/> before returning. Returns null when no server answers within the timeout.
        /// Throws <see cref="PipeOwnerMismatchException"/> (after closing the pipe) when the server is another account.
        /// </summary>
        public static NamedPipeClientStream? TryConnect(string pipeName, string expectedSid, TimeSpan timeout)
        {
            var client = new NamedPipeClientStream(
                ".",
                pipeName,
                PipeDirection.InOut,
                PipeOptions.Asynchronous,
                TokenImpersonationLevel.Identification);
            try
            {
                client.Connect(Math.Max(1, (int)timeout.TotalMilliseconds));
            }
            catch (Exception)
            {
                client.Dispose();
                return null;
            }

            if (!PipeServerIdentity.RunsAs(client, expectedSid))
            {
                client.Dispose();
                throw new PipeOwnerMismatchException();
            }

            return client;
        }

        /// <summary>
        /// Polls for the pipe until <paramref name="wait"/> has passed; returns null when it never appears. Each probe and
        /// each pause is cut short so the total never exceeds <paramref name="wait"/>.
        /// </summary>
        public static NamedPipeClientStream? WaitForPipe(string pipeName, string expectedSid, TimeSpan wait)
        {
            var watch = Stopwatch.StartNew();
            while (true)
            {
                var left = wait - watch.Elapsed;
                if (left <= TimeSpan.Zero)
                {
                    return null;
                }

                var client = TryConnect(pipeName, expectedSid, Min(ProbeTimeout, left));
                if (client != null)
                {
                    return client;
                }

                left = wait - watch.Elapsed;
                if (left <= TimeSpan.Zero)
                {
                    return null;
                }

                Thread.Sleep(Min(PollInterval, left));
            }
        }

        /// <summary>
        /// Sends one request line and reads one response line within <paramref name="timeout"/>. The result is null
        /// unless the host sent a complete LF-terminated line; callers must still check the envelope. Throws
        /// <see cref="TimeoutException"/> on timeout.
        /// </summary>
        public static Task<string?> ExchangeAsync(NamedPipeClientStream client, string requestLine, TimeSpan timeout) =>
            TimedIo.RunAsync(client, timeout, async () =>
            {
                var bytes = Utf8Text.Encode(requestLine + "\n");
                await client.WriteAsync(bytes, 0, bytes.Length).ConfigureAwait(false);
                await client.FlushAsync().ConfigureAwait(false);

                var read = await LineReader.ReadAsync(client, MaxResponseBytes).ConfigureAwait(false);
                if (read.Kind != LineReadKind.Line || !read.Terminated)
                {
                    return (string?)null;
                }

                return Utf8Text.TryDecode(read.Bytes, out var line) ? line : null;
            });

        private static TimeSpan Min(TimeSpan first, TimeSpan second) => first < second ? first : second;
    }
}
