using System;
using System.Threading;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>A unit of work queued on an instance worker.</summary>
    internal interface IWorkItem
    {
        /// <summary>Moves a queued item to running. False when its caller already gave up.</summary>
        bool TryStart();

        /// <summary>Runs the body on the worker thread. Never throws.</summary>
        void Run();

        /// <summary>Completes the caller's task with the result or the exception of <see cref="Run"/>.</summary>
        void Publish();

        /// <summary>Fails the item when it is still queued (the worker stopped).</summary>
        void FailIfQueued(Exception reason);
    }

    /// <summary>
    /// A queued call with its result. A caller that stops waiting (deadline, cancellation) calls
    /// <see cref="TryAbandon"/>; the worker then skips the item when it is still queued.
    /// </summary>
    internal sealed class WorkItem<T> : IWorkItem
    {
        private const int Queued = 0;
        private const int Running = 1;
        private const int Abandoned = 2;

        private readonly Func<T> _body;
        private readonly Action? _afterRun;
        private readonly TaskCompletionSource<T> _completion =
            new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);

        private int _state;
        private T? _value;
        private Exception? _error;

        /// <summary>Creates the item. <paramref name="afterRun"/> runs on the worker after the body, whatever it did.</summary>
        public WorkItem(Func<T> body, Action? afterRun)
        {
            _body = body ?? throw new ArgumentNullException(nameof(body));
            _afterRun = afterRun;
        }

        /// <summary>Completes when the body has run and been published.</summary>
        public Task<T> Task => _completion.Task;

        /// <summary>Marks the item abandoned if it has not started. Returns true when it was still queued.</summary>
        public bool TryAbandon() => Interlocked.CompareExchange(ref _state, Abandoned, Queued) == Queued;

        /// <inheritdoc />
        public bool TryStart() => Interlocked.CompareExchange(ref _state, Running, Queued) == Queued;

        /// <inheritdoc />
        public void Run()
        {
            try
            {
                _value = _body();
            }
            catch (Exception error)
            {
                _error = error;
            }

            _afterRun?.Invoke();
        }

        /// <inheritdoc />
        public void Publish()
        {
            if (_error != null)
            {
                _completion.TrySetException(_error);
                return;
            }

            _completion.TrySetResult(_value!);
        }

        /// <inheritdoc />
        public void FailIfQueued(Exception reason)
        {
            if (TryAbandon())
            {
                _completion.TrySetException(reason);
            }
        }
    }
}
