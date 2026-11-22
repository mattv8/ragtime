using System;
using System.Threading;
using System.Threading.Tasks;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>How a wait ended.</summary>
    internal enum WaitOutcome
    {
        /// <summary>The awaited task finished.</summary>
        Completed,

        /// <summary>The deadline passed first.</summary>
        TimedOut,

        /// <summary>The caller's cancellation token fired first.</summary>
        Cancelled,
    }

    /// <summary>Waits on the caller's thread for a task, the remaining deadline, or cancellation. Never blocks.</summary>
    internal static class DeadlineWait
    {
        /// <summary>Waits until <paramref name="work"/> completes, the deadline's remaining time passes, or cancellation.</summary>
        public static Task<WaitOutcome> WaitAsync(Task work, Deadline deadline, CancellationToken cancellationToken) =>
            WaitAsync(work, deadline.Remaining, cancellationToken);

        /// <summary>Waits until <paramref name="work"/> completes, <paramref name="limit"/> passes, or cancellation.</summary>
        public static async Task<WaitOutcome> WaitAsync(Task work, TimeSpan limit, CancellationToken cancellationToken)
        {
            var cancelled = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            using (cancellationToken.Register(() => cancelled.TrySetResult(true)))
            {
                var timeout = Task.Delay(limit < TimeSpan.Zero ? TimeSpan.Zero : limit);
                await Task.WhenAny(work, timeout, cancelled.Task).ConfigureAwait(false);
                if (work.IsCompleted)
                {
                    return WaitOutcome.Completed;
                }

                return cancelled.Task.IsCompleted ? WaitOutcome.Cancelled : WaitOutcome.TimedOut;
            }
        }
    }
}
