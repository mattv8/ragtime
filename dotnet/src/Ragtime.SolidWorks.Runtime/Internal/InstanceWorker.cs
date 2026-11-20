using System;
using System.Collections.Generic;
using System.Threading;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>
    /// One dedicated STA thread with a FIFO work queue. Everything SolidWorks-related for one instance runs here.
    /// Lock order: the manager lock may be taken before the gate of this class, never the reverse.
    /// </summary>
    internal sealed class InstanceWorker
    {
        private readonly object _gate = new object();
        private readonly Queue<IWorkItem> _queue = new Queue<IWorkItem>();
        private readonly Thread _thread;
        private int _pending;
        private bool _stopping;

        /// <summary>Creates the worker; call <see cref="Start"/> to run it.</summary>
        public InstanceWorker(string name)
        {
            _thread = new Thread(Loop) { IsBackground = true, Name = name };
            _thread.SetApartmentState(ApartmentState.STA);
        }

        /// <summary>Runs on the worker thread when it exits (releases COM references). Set before <see cref="Start"/>.</summary>
        public Action? ExitCallback { get; set; }

        /// <summary>Starts the thread.</summary>
        public void Start() => _thread.Start();

        /// <summary>True while work is queued or running.</summary>
        public bool HasPending
        {
            get
            {
                lock (_gate)
                {
                    return _pending > 0;
                }
            }
        }

        /// <summary>Queues an item. False when the worker is stopping.</summary>
        public bool TryEnqueue(IWorkItem item)
        {
            lock (_gate)
            {
                if (_stopping)
                {
                    return false;
                }

                _queue.Enqueue(item);
                _pending++;
                Monitor.PulseAll(_gate);
                return true;
            }
        }

        /// <summary>Stops accepting work, fails the queued items, and lets the running item finish.</summary>
        public void Stop(Exception reason)
        {
            lock (_gate)
            {
                _stopping = true;
            }

            FailQueued(reason);
        }

        /// <summary>Fails the queued items but keeps the worker running (used before a final close item).</summary>
        public void FailQueued(Exception reason)
        {
            List<IWorkItem> drained;
            lock (_gate)
            {
                drained = new List<IWorkItem>(_queue);
                _queue.Clear();
                _pending -= drained.Count;
                Monitor.PulseAll(_gate);
            }

            foreach (var item in drained)
            {
                item.FailIfQueued(reason);
            }
        }

        private void Loop()
        {
            try
            {
                while (TryTakeNext(out var item))
                {
                    var started = item.TryStart();
                    if (started)
                    {
                        item.Run();
                    }

                    lock (_gate)
                    {
                        _pending--;
                    }

                    if (started)
                    {
                        item.Publish();
                    }
                }
            }
            finally
            {
                ExitCallback?.Invoke();
            }
        }

        private bool TryTakeNext(out IWorkItem item)
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
