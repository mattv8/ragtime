using System;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Security.Principal;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Proves that the server end of a connected client pipe runs as the expected account. Used before any request byte
    /// is written, so a squatting process never receives a request. Any failure counts as "not the expected account".
    /// </summary>
    internal static class PipeServerIdentity
    {
        private const uint ProcessQueryLimitedInformation = 0x00001000;
        private const uint TokenQuery = 0x00000008;

        /// <summary>True only when the server process's token user SID equals <paramref name="expectedSid"/>.</summary>
        public static bool RunsAs(NamedPipeClientStream client, string expectedSid)
        {
            try
            {
                var pipe = client.SafePipeHandle.DangerousGetHandle();
                if (!GetNamedPipeServerProcessId(pipe, out var processId) || processId == 0)
                {
                    return false;
                }

                var process = OpenProcess(ProcessQueryLimitedInformation, false, processId);
                if (process == IntPtr.Zero)
                {
                    return false;
                }

                try
                {
                    var sid = UserSidOfProcess(process);
                    return string.Equals(sid, expectedSid, StringComparison.OrdinalIgnoreCase);
                }
                finally
                {
                    CloseHandle(process);
                }
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static string? UserSidOfProcess(IntPtr process)
        {
            if (!OpenProcessToken(process, TokenQuery, out var token))
            {
                return null;
            }

            try
            {
                using (var identity = new WindowsIdentity(token))
                {
                    return identity.User?.Value;
                }
            }
            finally
            {
                CloseHandle(token);
            }
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetNamedPipeServerProcessId(IntPtr pipe, out uint serverProcessId);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr OpenProcess(uint desiredAccess, [MarshalAs(UnmanagedType.Bool)] bool inheritHandle, uint processId);

        [DllImport("advapi32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool OpenProcessToken(IntPtr processHandle, uint desiredAccess, out IntPtr tokenHandle);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr handle);
    }
}
