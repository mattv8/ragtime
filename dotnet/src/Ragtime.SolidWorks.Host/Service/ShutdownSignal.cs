using System.Threading;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>One-shot "shut down" signal, raised by host.shutdown or by a console stop event.</summary>
    internal sealed class ShutdownSignal
    {
        private readonly TaskCompletionSource<bool> _requested =
            new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        /// <summary>Raises the signal. Safe to call more than once.</summary>
        public void Request()
        {
            _requested.TrySetResult(true);
        }

        /// <summary>Signal shape required by the request processor's shutdown callback.</summary>
        public Task RequestAsync(CancellationToken cancellationToken)
        {
            Request();
            return Task.CompletedTask;
        }

        /// <summary>Blocks the calling thread until the signal is raised.</summary>
        public void Wait()
        {
            _requested.Task.GetAwaiter().GetResult();
        }
    }
}
