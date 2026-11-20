using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>
    /// Plain Win32 window helpers for host-launched SolidWorks instances. Runs on the caller's thread; no COM.
    /// Only ever applied to owned instances: attached instances are never touched.
    /// </summary>
    internal static class WindowControl
    {
        /// <summary>SW_HIDE: hides the window (ShowWindow nCmdShow value).</summary>
        private const int SwHide = 0;

        private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool IsWindowVisible(IntPtr hWnd);

        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern int GetWindowTextLength(IntPtr hWnd);

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        /// <summary>
        /// The visible top-level windows of one process that have a non-empty title. Enumerates every top-level window
        /// of the desktop and keeps the ones owned by the process id.
        /// </summary>
        public static IReadOnlyList<IntPtr> VisibleTitledWindows(int processId)
        {
            var found = new List<IntPtr>();
            var target = (uint)processId;

            // The delegate is a local that stays referenced until EnumWindows returns (KeepAlive below).
            EnumWindowsProc callback = (hWnd, lParam) =>
            {
                try
                {
                    GetWindowThreadProcessId(hWnd, out var owner);
                    if (owner == target && IsWindowVisible(hWnd) && GetWindowTextLength(hWnd) > 0)
                    {
                        found.Add(hWnd);
                    }
                }
                catch (Exception)
                {
                    // A window that cannot be read is skipped; the enumeration continues.
                }

                return true;
            };

            if (!EnumWindows(callback, IntPtr.Zero))
            {
                var error = Marshal.GetLastWin32Error();
                if (error != 0)
                {
                    throw new System.ComponentModel.Win32Exception(error, "EnumWindows failed.");
                }
            }

            GC.KeepAlive(callback);
            return found;
        }

        /// <summary>Hides one top-level window (ShowWindow SW_HIDE). Returns false when the window call reported failure.</summary>
        public static bool Hide(IntPtr hWnd) => ShowWindow(hWnd, SwHide);
    }
}
