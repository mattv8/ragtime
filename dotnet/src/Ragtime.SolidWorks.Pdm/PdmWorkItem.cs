using System;
using System.Threading.Tasks;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>What happened when a caller gave up waiting for a work item.</summary>
    internal enum PdmAbandonOutcome
    {
        /// <summary>The item was still queued and now never runs.</summary>
        Cancelled,

        /// <summary>The item is running; the PDM thread stays busy until it returns.</summary>
        Stuck,

        /// <summary>The item had already finished; its result can still be awaited.</summary>
        Done,
    }

    /// <summary>A unit of work queued on the PDM thread, seen by the worker without its type.</summary>
    internal interface IPdmWorkItem
    {
        /// <summary>True when the caller gave up while the item was running.</summary>
        bool TimedOut { get; }

        /// <summary>Moves a queued item to running. False when it was cancelled or failed.</summary>
        bool TryStart();

        /// <summary>Runs the body on the PDM thread. Never throws.</summary>
        void Run();

        /// <summary>Completes the caller's task with the result or the exception of <see cref="Run"/>.</summary>
        void Publish();

        /// <summary>Fails the item when it is still queued (the worker is stopping).</summary>
        void FailIfQueued(Exception reason);
    }

    /// <summary>
    /// A queued call and its result. State changes only under the item's own lock, so a caller that gives up and the
    /// worker that starts or finishes the item never disagree about whether it is running.
    /// </summary>
    internal sealed class PdmWorkItem<T> : IPdmWorkItem
    {
        private enum State
        {
            Queued,
            Running,
            Finished,
            Abandoned,
        }

        private readonly object _gate = new object();
        private readonly Func<T> _body;
        private readonly TaskCompletionSource<T> _completion =
            new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);

        private State _state = State.Queued;
        private bool _timedOut;
        private T? _value;
        private Exception? _error;
        private readonly Action<Exception?>? _lateCompletion;

        /// <summary>Creates the item around the body that runs on the PDM thread.</summary>
        public PdmWorkItem(Func<T> body, Action<Exception?>? lateCompletion = null)
        {
            _body = body;
            _lateCompletion = lateCompletion;
        }

        /// <summary>Completes when the body has run and been published.</summary>
        public Task<T> Task => _completion.Task;

        /// <inheritdoc />
        public bool TimedOut
        {
            get
            {
                lock (_gate)
                {
                    return _timedOut;
                }
            }
        }

        /// <summary>Gives up on the item. A queued item is cancelled; a running one is marked stuck.</summary>
        public PdmAbandonOutcome Abandon()
        {
            lock (_gate)
            {
                switch (_state)
                {
                    case State.Queued:
                        _state = State.Abandoned;
                        return PdmAbandonOutcome.Cancelled;
                    case State.Running:
                        _timedOut = true;
                        return PdmAbandonOutcome.Stuck;
                    default:
                        return PdmAbandonOutcome.Done;
                }
            }
        }

        /// <inheritdoc />
        public bool TryStart()
        {
            lock (_gate)
            {
                if (_state != State.Queued)
                {
                    return false;
                }

                _state = State.Running;
                return true;
            }
        }

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

            bool late;
            lock (_gate)
            {
                late = _timedOut;
                _state = State.Finished;
            }
            if (late)
            {
                try { _lateCompletion?.Invoke(_error); }
                catch (Exception) { }
            }
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
            var queued = false;
            lock (_gate)
            {
                if (_state == State.Queued)
                {
                    _state = State.Abandoned;
                    queued = true;
                }
            }

            if (queued)
            {
                _completion.TrySetException(reason);
            }
        }
    }
}
