using System;
using System.Collections.Generic;
using System.Threading;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>
    /// One dedicated STA thread with a FIFO queue. Every PDM COM object is created and used on it. When a caller gives
    /// up on a running item the worker reports busy until that item returns; a queued item that is given up never runs.
    /// Lock order: the worker lock may be held while reading an item's state, never the reverse.
    /// </summary>
    internal sealed class PdmWorker
    {
        private readonly object _gate = new object();
        private readonly Queue<IPdmWorkItem> _queue = new Queue<IPdmWorkItem>();
        private readonly Action _onExit;
        private Thread? _thread;
        private IPdmWorkItem? _current;
        private bool _stopping;

        /// <summary>Creates the worker. <paramref name="onExit"/> runs on the PDM thread when the thread ends.</summary>
        public PdmWorker(Action onExit)
        {
            _onExit = onExit;
        }

        /// <summary>True while an item that its caller gave up on is still running.</summary>
        public bool IsBusy
        {
            get
            {
                lock (_gate)
                {
                    return _current != null && _current.TimedOut;
                }
            }
        }

        /// <summary>Queues work for the PDM thread, starting that thread on first use.</summary>
        /// <exception cref="PdmGatewayException">Busy while an earlier item that its caller gave up on still runs.</exception>
        public PdmWorkItem<T> Enqueue<T>(Func<T> body, Action<Exception?>? lateCompletion = null)
        {
            var item = new PdmWorkItem<T>(body, lateCompletion);
            lock (_gate)
            {
                if (_stopping)
                {
                    throw new ObjectDisposedException(nameof(PdmGateway));
                }

                if (_current != null && _current.TimedOut)
                {
                    throw PdmErrors.Busy();
                }

                if (_thread == null)
                {
                    StartThread();
                }

                _queue.Enqueue(item);
                Monitor.PulseAll(_gate);
            }

            return item;
        }

        /// <summary>Stops accepting work and fails the queued items. The running item finishes first.</summary>
        public void Stop()
        {
            List<IPdmWorkItem> drained;
            lock (_gate)
            {
                _stopping = true;
                drained = new List<IPdmWorkItem>(_queue);
                _queue.Clear();
                Monitor.PulseAll(_gate);
            }

            foreach (var item in drained)
            {
                item.FailIfQueued(new ObjectDisposedException(nameof(PdmGateway)));
            }
        }

        private void StartThread()
        {
            _thread = new Thread(Loop) { IsBackground = true, Name = "Ragtime PDM worker" };
            _thread.SetApartmentState(ApartmentState.STA);
            _thread.Start();
        }

        private void Loop()
        {
            try
            {
                while (TakeNext(out var item))
                {
                    if (!item.TryStart())
                    {
                        continue;
                    }

                    lock (_gate)
                    {
                        _current = item;
                    }

                    item.Run();

                    lock (_gate)
                    {
                        _current = null;
                    }

                    item.Publish();
                }
            }
            finally
            {
                _onExit();
            }
        }

        private bool TakeNext(out IPdmWorkItem item)
        {
            lock (_gate)
            {
                while (_queue.Count == 0 && !_stopping)
                {
                    Monitor.Wait(_gate);
                }

                if (_queue.Count == 0)
                {
                    item = null!;
                    return false;
                }

                item = _queue.Dequeue();
                return true;
            }
        }
    }
}
