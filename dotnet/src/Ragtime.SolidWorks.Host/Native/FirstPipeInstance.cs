using System;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using Microsoft.Win32.SafeHandles;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Creates the first listener of the host pipe with FILE_FLAG_FIRST_PIPE_INSTANCE. .NET Framework 4.8 has no
    /// PipeOptions.FirstPipeInstance, so the handle is created here and wrapped in a NamedPipeServerStream. The call
    /// fails when any instance of the name already exists, which is how a squatting process is detected.
    /// </summary>
    internal static class FirstPipeInstance
    {
        private const uint PipeAccessDuplex = 0x00000003;
        private const uint FileFlagOverlapped = 0x40000000;
        private const uint FileFlagFirstPipeInstance = 0x00080000;
        private const uint PipeTypeByteReadByteWait = 0x00000000;
        private const uint PipeUnlimitedInstances = 255;

        /// <summary>Creates the first server instance; throws <see cref="PipeNameInUseException"/> when the name is taken.</summary>
        public static NamedPipeServerStream Create(string pipeName, PipeSecurity security, int bufferBytes)
        {
            var descriptor = security.GetSecurityDescriptorBinaryForm();
            var pinned = GCHandle.Alloc(descriptor, GCHandleType.Pinned);
            try
            {
                var attributes = new SecurityAttributes
                {
                    nLength = Marshal.SizeOf(typeof(SecurityAttributes)),
                    lpSecurityDescriptor = pinned.AddrOfPinnedObject(),
                    bInheritHandle = 0,
                };

                var handle = CreateNamedPipe(
                    @"\\.\pipe\" + pipeName,
                    PipeAccessDuplex | FileFlagOverlapped | FileFlagFirstPipeInstance,
                    PipeTypeByteReadByteWait,
                    PipeUnlimitedInstances,
                    (uint)bufferBytes,
                    (uint)bufferBytes,
                    0,
                    ref attributes);
                if (handle == new IntPtr(-1))
                {
                    throw new PipeNameInUseException("CreateNamedPipe failed with Win32 error " + Marshal.GetLastWin32Error() + ".");
                }

                return new NamedPipeServerStream(PipeDirection.InOut, true, false, new SafePipeHandle(handle, true));
            }
            finally
            {
                pinned.Free();
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct SecurityAttributes
        {
            public int nLength;
            public IntPtr lpSecurityDescriptor;
            public int bInheritHandle;
        }

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr CreateNamedPipe(
            string name,
            uint openMode,
            uint pipeMode,
            uint maxInstances,
            uint outBufferSize,
            uint inBufferSize,
            uint defaultTimeout,
            ref SecurityAttributes securityAttributes);
    }
}
