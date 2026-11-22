using System;
using System.Runtime.InteropServices;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>Console window and control handling for <c>serve</c>. Win32 calls only.</summary>
    internal static class ConsoleControl
    {
        private const int SwHide = 0;

        // Kept alive for the life of the process: the OS calls it from another thread.
        private static ConsoleCtrlHandler? _handler;

        /// <summary>Hides the console window of this process, if it has one (the task runs hidden).</summary>
        public static void HideWindow()
        {
            var window = GetConsoleWindow();
            if (window != IntPtr.Zero)
            {
                ShowWindow(window, SwHide);
            }
        }

        /// <summary>
        /// Calls <paramref name="onStop"/> on Ctrl+C, console close, logoff and shutdown. The handler returns true so
        /// the default termination does not run; the caller then shuts down in order. Returns false when the
        /// handler could not be registered.
        /// </summary>
        public static bool RegisterStopHandler(Action onStop)
        {
            _handler = ctrlType =>
            {
                onStop();
                return true;
            };

            return SetConsoleCtrlHandler(_handler, true);
        }

        private delegate bool ConsoleCtrlHandler(int ctrlType);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetConsoleWindow();

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetConsoleCtrlHandler(ConsoleCtrlHandler handler, [MarshalAs(UnmanagedType.Bool)] bool add);
    }
}
