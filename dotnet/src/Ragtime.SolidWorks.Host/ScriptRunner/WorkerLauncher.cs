using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.Win32.SafeHandles;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Starts the script worker suspended, so it is inside its Job Object before it runs any code, then resumes it. Its
    /// standard output and error are this run's <c>worker.out</c> handle; its standard input is NUL. A handle list limits
    /// what it inherits to those two handles, so no other run's handles leak into it.
    /// </summary>
    internal static class WorkerLauncher
    {
        private const uint GenericRead = 0x80000000;
        private const uint GenericWrite = 0x40000000;
        private const uint FileShareRead = 0x1;
        private const uint FileShareWrite = 0x2;
        private const uint CreateAlways = 2;
        private const uint OpenExisting = 3;
        private const uint FileAttributeNormal = 0x80;
        private const uint CreateSuspendedFlag = 0x4;
        private const uint CreateNoWindowFlag = 0x08000000;
        private const uint ExtendedStartupInfoPresent = 0x00080000;
        private const int StartfUseStdHandles = 0x100;
        private const uint WaitObject0 = 0;
        private const uint WaitFailed = 0xFFFFFFFF;
        private const uint ResumeFailed = uint.MaxValue;
        private const uint TerminatedExitCode = 1;
        private const int PollMilliseconds = 200;
        private const string NulDevice = "NUL";
        private static readonly IntPtr InvalidHandleValue = new IntPtr(-1);

        /// <summary>
        /// Starts <c>&lt;hostExe&gt; script-worker --run-folder "&lt;folder&gt;"</c> suspended with no window, assigns it to
        /// <paramref name="job"/>, then resumes it. The output and input handles are closed in this process right after
        /// <c>CreateProcess</c>, on every path. Returns the process handle, which the caller owns. On a failure after
        /// creation the process is terminated and nothing is left open.
        /// </summary>
        public static SafeKernelHandle Start(string hostExePath, string folder, string outputPath, SafeJobHandle job)
        {
            var information = StartSuspended(hostExePath, folder, outputPath);
            var process = new SafeKernelHandle(information.hProcess);
            var thread = new SafeKernelHandle(information.hThread);
            try
            {
                JobObject.Assign(job, process);
                if (ResumeThread(thread) == ResumeFailed)
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread failed.");
                }

                return process;
            }
            catch (Exception)
            {
                TerminateProcess(process, TerminatedExitCode);
                process.Dispose();
                throw;
            }
            finally
            {
                thread.Dispose();
            }
        }

        /// <summary>True when the process exited; false when the timeout passed or the call was cancelled. Throws when the wait itself fails.</summary>
        public static bool WaitForExit(SafeKernelHandle process, TimeSpan timeout, CancellationToken cancellationToken)
        {
            var elapsed = Stopwatch.StartNew();
            while (true)
            {
                var result = WaitForSingleObject(process, PollMilliseconds);
                if (result == WaitObject0)
                {
                    return true;
                }

                if (result == WaitFailed)
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "WaitForSingleObject failed.");
                }

                if (cancellationToken.IsCancellationRequested || elapsed.Elapsed >= timeout)
                {
                    return false;
                }
            }
        }

        /// <summary>The exit code of a process that has exited.</summary>
        public static int ExitCode(SafeKernelHandle process)
        {
            if (!GetExitCodeProcess(process, out var code))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "GetExitCodeProcess failed.");
            }

            return unchecked((int)code);
        }

        /// <summary>Opens the output and input handles, creates the process with exactly those inherited, and closes them.</summary>
        private static PROCESS_INFORMATION StartSuspended(string hostExePath, string folder, string outputPath)
        {
            using (var output = OpenInheritable(outputPath, GenericWrite, FileShareRead, CreateAlways))
            using (var input = OpenInheritable(NulDevice, GenericRead, FileShareRead | FileShareWrite, OpenExisting))
            using (var inherited = new HandleList(new[] { output.DangerousGetHandle(), input.DangerousGetHandle() }))
            {
                var commandLine = new StringBuilder(
                    "\"" + hostExePath + "\" " + CommandNames.ScriptWorker + " --run-folder \"" + folder + "\"");
                var startup = new STARTUPINFOEX
                {
                    StartupInfo = new STARTUPINFO
                    {
                        dwFlags = StartfUseStdHandles,
                        hStdInput = input.DangerousGetHandle(),
                        hStdOutput = output.DangerousGetHandle(),
                        hStdError = output.DangerousGetHandle(),
                    },
                    lpAttributeList = inherited.Pointer,
                };
                startup.StartupInfo.cb = Marshal.SizeOf(typeof(STARTUPINFOEX));
                var flags = CreateSuspendedFlag | CreateNoWindowFlag | ExtendedStartupInfoPresent;
                if (!CreateProcess(null, commandLine, IntPtr.Zero, IntPtr.Zero, true, flags, IntPtr.Zero, null, ref startup, out var information))
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateProcess failed.");
                }

                return information;
            }
        }

        private static SafeKernelHandle OpenInheritable(string path, uint access, uint share, uint disposition)
        {
            var attributes = new SECURITY_ATTRIBUTES
            {
                nLength = (uint)Marshal.SizeOf(typeof(SECURITY_ATTRIBUTES)),
                bInheritHandle = true,
            };
            var handle = CreateFile(path, access, share, ref attributes, disposition, FileAttributeNormal, IntPtr.Zero);
            if (handle == InvalidHandleValue)
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateFile failed: " + path);
            }

            return new SafeKernelHandle(handle);
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CreateProcess(
            string? application,
            StringBuilder commandLine,
            IntPtr processAttributes,
            IntPtr threadAttributes,
            bool inheritHandles,
            uint flags,
            IntPtr environment,
            string? directory,
            ref STARTUPINFOEX startup,
            out PROCESS_INFORMATION information);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateFile(
            string path,
            uint access,
            uint share,
            ref SECURITY_ATTRIBUTES attributes,
            uint disposition,
            uint flags,
            IntPtr template);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint ResumeThread(SafeKernelHandle thread);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateProcess(SafeKernelHandle process, uint exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint WaitForSingleObject(SafeKernelHandle handle, int milliseconds);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetExitCodeProcess(SafeKernelHandle process, out uint exitCode);
    }

    /// <summary>
    /// A <c>PROC_THREAD_ATTRIBUTE_HANDLE_LIST</c> attribute list. The worker inherits only the listed handles. Dispose it
    /// after <c>CreateProcess</c> returns.
    /// </summary>
    internal sealed class HandleList : IDisposable
    {
        private const uint ProcThreadAttributeHandleList = 0x00020002;

        private IntPtr _list = IntPtr.Zero;
        private IntPtr _handles = IntPtr.Zero;
        private bool _initialized;

        public HandleList(IntPtr[] handles)
        {
            try
            {
                var size = IntPtr.Zero;
                InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref size);
                _list = Marshal.AllocHGlobal(size);
                if (!InitializeProcThreadAttributeList(_list, 1, 0, ref size))
                {
                    throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "InitializeProcThreadAttributeList failed.");
                }

                _initialized = true;
                _handles = Marshal.AllocHGlobal(IntPtr.Size * handles.Length);
                Marshal.Copy(handles, 0, _handles, handles.Length);
                if (!UpdateProcThreadAttribute(_list, 0, (IntPtr)ProcThreadAttributeHandleList, _handles, (IntPtr)(IntPtr.Size * handles.Length), IntPtr.Zero, IntPtr.Zero))
                {
                    throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "UpdateProcThreadAttribute failed.");
                }
            }
            catch (Exception)
            {
                Dispose();
                throw;
            }
        }

        /// <summary>The attribute list, for <c>STARTUPINFOEX.lpAttributeList</c>.</summary>
        public IntPtr Pointer => _list;

        /// <inheritdoc />
        public void Dispose()
        {
            if (_initialized)
            {
                DeleteProcThreadAttributeList(_list);
                _initialized = false;
            }

            if (_list != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(_list);
                _list = IntPtr.Zero;
            }

            if (_handles != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(_handles);
                _handles = IntPtr.Zero;
            }
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, int flags, ref IntPtr size);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, IntPtr attribute, IntPtr value, IntPtr size, IntPtr previous, IntPtr returnSize);

        [DllImport("kernel32.dll")]
        private static extern void DeleteProcThreadAttributeList(IntPtr list);
    }

    /// <summary>A kernel handle that is closed on dispose.</summary>
    internal sealed class SafeKernelHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public SafeKernelHandle(IntPtr handle)
            : base(true)
        {
            SetHandle(handle);
        }

        protected override bool ReleaseHandle()
        {
            return CloseHandle(handle);
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    internal struct STARTUPINFO
    {
        public int cb;
        public string? lpReserved;
        public string? lpDesktop;
        public string? lpTitle;
        public int dwX;
        public int dwY;
        public int dwXSize;
        public int dwYSize;
        public int dwXCountChars;
        public int dwYCountChars;
        public int dwFillAttribute;
        public int dwFlags;
        public short wShowWindow;
        public short cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    internal struct STARTUPINFOEX
    {
        public STARTUPINFO StartupInfo;
        public IntPtr lpAttributeList;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct PROCESS_INFORMATION
    {
        public IntPtr hProcess;
        public IntPtr hThread;
        public uint dwProcessId;
        public uint dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct SECURITY_ATTRIBUTES
    {
        public uint nLength;
        public IntPtr lpSecurityDescriptor;
        [MarshalAs(UnmanagedType.Bool)]
        public bool bInheritHandle;
    }
}
