using System;
using System.Runtime.InteropServices;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Best-effort check whether the current user has a session other than session 0 (Terminal Services API).
    /// Used only to explain host_not_running; never used to make decisions.
    /// </summary>
    internal static class UserSessionProbe
    {
        private const int WtsUserName = 5;
        private const int WtsDomainName = 7;

        /// <summary>True when a non-zero session belongs to the current user; false when none does; null when unknown.</summary>
        public static bool? HasSessionOtherThanZero()
        {
            try
            {
                if (!WTSEnumerateSessions(IntPtr.Zero, 0, 1, out var sessions, out var count))
                {
                    return null;
                }

                try
                {
                    return AnySessionOfCurrentUser(sessions, count);
                }
                finally
                {
                    WTSFreeMemory(sessions);
                }
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static bool AnySessionOfCurrentUser(IntPtr sessions, int count)
        {
            var size = Marshal.SizeOf(typeof(WtsSessionInfo));
            for (var index = 0; index < count; index++)
            {
                var entry = (WtsSessionInfo)Marshal.PtrToStructure(IntPtr.Add(sessions, index * size), typeof(WtsSessionInfo));
                if (entry.SessionId != 0 && BelongsToCurrentUser(entry.SessionId))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool BelongsToCurrentUser(int sessionId)
        {
            var userName = QueryString(sessionId, WtsUserName);
            var domain = QueryString(sessionId, WtsDomainName);
            return userName.Length > 0
                && string.Equals(userName, Environment.UserName, StringComparison.OrdinalIgnoreCase)
                && string.Equals(domain, Environment.UserDomainName, StringComparison.OrdinalIgnoreCase);
        }

        private static string QueryString(int sessionId, int infoClass)
        {
            if (!WTSQuerySessionInformation(IntPtr.Zero, sessionId, infoClass, out var buffer, out _))
            {
                return string.Empty;
            }

            try
            {
                return Marshal.PtrToStringUni(buffer) ?? string.Empty;
            }
            finally
            {
                WTSFreeMemory(buffer);
            }
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct WtsSessionInfo
        {
            public int SessionId;

            [MarshalAs(UnmanagedType.LPWStr)]
            public string WinStationName;

            public int State;
        }

        [DllImport("wtsapi32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool WTSEnumerateSessions(IntPtr server, int reserved, int version, out IntPtr sessions, out int count);

        [DllImport("wtsapi32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool WTSQuerySessionInformation(IntPtr server, int sessionId, int infoClass, out IntPtr buffer, out int bytesReturned);

        [DllImport("wtsapi32.dll")]
        private static extern void WTSFreeMemory(IntPtr memory);
    }
}
