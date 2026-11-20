using System;
using System.Diagnostics;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Monotonic millisecond clock. Injected so deadline arithmetic can be tested without sleeping.</summary>
    public interface IMonotonicClock
    {
        /// <summary>Milliseconds since an arbitrary fixed origin; never goes backwards.</summary>
        long ElapsedMilliseconds { get; }
    }

    /// <summary>Clock backed by <see cref="Stopwatch"/>.</summary>
    public sealed class SystemMonotonicClock : IMonotonicClock
    {
        private readonly Stopwatch _watch = Stopwatch.StartNew();

        /// <inheritdoc />
        public long ElapsedMilliseconds => _watch.ElapsedMilliseconds;
    }

    /// <summary>
    /// The time a request may spend, started when the object is created. The host uses it for the request
    /// deadline; the processor and the instance manager read <see cref="Remaining"/>.
    /// </summary>
    public sealed class Deadline
    {
        private readonly IMonotonicClock _clock;
        private readonly long _startMs;
        private readonly long _totalMs;

        /// <summary>Starts a deadline of <paramref name="seconds"/> (at least 1) on the given clock.</summary>
        public Deadline(int seconds, IMonotonicClock clock)
        {
            if (seconds < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(seconds), "A deadline is at least one second.");
            }

            _clock = clock ?? throw new ArgumentNullException(nameof(clock));
            _totalMs = seconds * 1000L;
            _startMs = clock.ElapsedMilliseconds;
        }

        /// <summary>The whole deadline.</summary>
        public TimeSpan Total => TimeSpan.FromMilliseconds(_totalMs);

        /// <summary>Time left, never negative.</summary>
        public TimeSpan Remaining
        {
            get
            {
                var left = _totalMs - (_clock.ElapsedMilliseconds - _startMs);
                return TimeSpan.FromMilliseconds(left > 0 ? left : 0);
            }
        }

        /// <summary>True when nothing is left.</summary>
        public bool Expired => Remaining == TimeSpan.Zero;
    }

    /// <summary>Client-side arithmetic of <c>call</c> (contract section 3).</summary>
    public static class CallTiming
    {
        /// <summary>
        /// Overall cap of one <c>call</c>: deadline_seconds + 5 seconds. Time spent starting or waiting for the
        /// host counts toward it.
        /// </summary>
        public static TimeSpan OverallCap(int deadlineSeconds) =>
            TimeSpan.FromSeconds(deadlineSeconds + ProtocolInfo.CallGraceSeconds);

        /// <summary>
        /// The deadline_seconds to forward to the host after <paramref name="spent"/> has passed: the original
        /// minus the spent time, rounded down to whole seconds. Returns null when less than one whole second is
        /// left; the caller then answers host_not_running without contacting the pipe.
        /// </summary>
        public static int? ForwardedDeadlineSeconds(int deadlineSeconds, TimeSpan spent)
        {
            var remainingMs = deadlineSeconds * 1000.0 - spent.TotalMilliseconds;
            if (remainingMs < 1000.0)
            {
                return null;
            }

            return (int)Math.Floor(remainingMs / 1000.0);
        }
    }
}
