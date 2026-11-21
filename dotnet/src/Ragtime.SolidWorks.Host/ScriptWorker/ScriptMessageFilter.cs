using System;
using System.Runtime.InteropServices;

namespace Ragtime.SolidWorks.Host
{
    [ComImport, Guid("00000016-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IScriptMessageFilter
    {
        [PreserveSig] int HandleInComingCall(int type, IntPtr task, int ticks, IntPtr info);
        [PreserveSig] int RetryRejectedCall(IntPtr task, int ticks, int reject);
        [PreserveSig] int MessagePending(IntPtr task, int ticks, int pending);
    }

    internal sealed class ScriptMessageFilter : IScriptMessageFilter
    {
        private readonly DateTime _deadline;
        public ScriptMessageFilter(DateTime deadline) => _deadline = deadline;
        public int HandleInComingCall(int type, IntPtr task, int ticks, IntPtr info) => 0;
        public int RetryRejectedCall(IntPtr task, int ticks, int reject)
        {
            var remaining = (_deadline - DateTime.UtcNow).TotalMilliseconds;
            var backoff = Math.Min(500, 50 + Math.Max(0, ticks) / 10);
            return (reject == 1 || reject == 2) && remaining > 0 ? (int)Math.Min(backoff, remaining) : -1;
        }
        public int MessagePending(IntPtr task, int ticks, int pending) => 2;

        [DllImport("ole32.dll")]
        private static extern int CoRegisterMessageFilter(IScriptMessageFilter? filter, out IScriptMessageFilter? previous);

        public static void Register(IScriptMessageFilter? filter, out IScriptMessageFilter? previous) =>
            Marshal.ThrowExceptionForHR(CoRegisterMessageFilter(filter, out previous));
    }
}
