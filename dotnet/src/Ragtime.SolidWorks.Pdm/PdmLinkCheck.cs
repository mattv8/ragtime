using System;
using System.Runtime.InteropServices;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>
    /// Detects symbolic links and junctions on one path component without following it. Core's LocalFileCheck keeps
    /// its own check private, so the PDM project queries the component itself.
    /// </summary>
    internal static class PdmLinkCheck
    {
        private const uint ReparsePointAttribute = 0x00000400;
        private const uint NameSurrogateTag = 0x20000000;
        private const int ErrorFileNotFound = 2;
        private const int ErrorPathNotFound = 3;
        private static readonly IntPtr InvalidFindHandle = new IntPtr(-1);

        /// <summary>
        /// True when the path exists and is a name-surrogate reparse point (symbolic link or junction). A path that does not
        /// exist is not a link. A path that cannot be queried counts as a link (fails closed).
        /// </summary>
        public static bool IsNameSurrogate(string path)
        {
            var handle = FindFirstFileW(path, out var data);
            if (handle == InvalidFindHandle)
            {
                var error = Marshal.GetLastWin32Error();
                return error != ErrorFileNotFound && error != ErrorPathNotFound;
            }

            FindClose(handle);
            return (data.dwFileAttributes & ReparsePointAttribute) != 0 && (data.dwReserved0 & NameSurrogateTag) != 0;
        }

        /// <summary>WIN32_FIND_DATAW. For a reparse point, dwReserved0 holds the reparse tag.</summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct WIN32_FIND_DATAW
        {
            public uint dwFileAttributes;
            public uint ftCreationTimeLow;
            public uint ftCreationTimeHigh;
            public uint ftLastAccessTimeLow;
            public uint ftLastAccessTimeHigh;
            public uint ftLastWriteTimeLow;
            public uint ftLastWriteTimeHigh;
            public uint nFileSizeHigh;
            public uint nFileSizeLow;
            public uint dwReserved0;
            public uint dwReserved1;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
            public string cFileName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 14)]
            public string cAlternateFileName;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr FindFirstFileW(string fileName, out WIN32_FIND_DATAW findData);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FindClose(IntPtr findFile);
    }
}
