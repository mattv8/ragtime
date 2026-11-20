using System;
using System.IO;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Bounds pipe I/O by time. Cancellation tokens do not reliably interrupt a pending pipe read on .NET Framework,
    /// so on timeout the stream is closed, which aborts the pending operation.
    /// </summary>
    internal static class TimedIo
    {
        /// <summary>Runs the operation; throws <see cref="TimeoutException"/> and closes the stream when it does not finish in time.</summary>
        public static async Task<T> RunAsync<T>(Stream stream, TimeSpan timeout, Func<Task<T>> operation)
        {
            var work = operation();
            var finished = await Task.WhenAny(work, Task.Delay(timeout < TimeSpan.Zero ? TimeSpan.Zero : timeout)).ConfigureAwait(false);
            if (!ReferenceEquals(finished, work))
            {
                _ = work.ContinueWith(ObserveFault, TaskContinuationOptions.OnlyOnFaulted);
                CloseQuietly(stream);
                throw new TimeoutException("The pipe did not complete the operation in time.");
            }

            return await work.ConfigureAwait(false);
        }

        /// <summary>Closes a stream and ignores errors.</summary>
        public static void CloseQuietly(Stream stream)
        {
            try
            {
                stream.Dispose();
            }
            catch (Exception)
            {
                // Already closed or aborted by the peer.
            }
        }

        private static void ObserveFault(Task faulted)
        {
            _ = faulted.Exception;
        }
    }
}
