using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Checks that a document path is on local storage before SolidWorks opens it. The drive of the path must be fixed,
    /// removable or a RAM disk; every path component is then queried without following it, and a symbolic link or
    /// junction (a name-surrogate reparse point) is rejected before anything could traverse it. The opened file's final
    /// path must also be local (defence in depth). Fails closed. Stateless.
    /// </summary>
    public static class LocalFileCheck
    {
        private const uint DriveRemovable = 2;
        private const uint DriveFixed = 3;
        private const uint DriveRamDisk = 6;
        private const uint FileReadAttributes = 0x0080;
        private const uint ShareReadWriteDelete = 0x0007;
        private const uint OpenExisting = 3;
        private const uint FileFlagBackupSemantics = 0x02000000;
        private const uint VolumeNameDos = 0;
        private const uint ReparsePointAttribute = 0x00000400;
        private const uint NameSurrogateTag = 0x20000000;
        private const int ErrorFileNotFound = 2;
        private const int ErrorPathNotFound = 3;
        private static readonly IntPtr InvalidFindHandle = new IntPtr(-1);
        private static readonly char[] ForbiddenInComponent = { '*', '?', ':', '<', '>', '"', '|' };
        private const string LocalOnlyMessage =
            "Only files on local drives are allowed. Mapped network drives and links to network locations are not.";
        private const string LinkMessage =
            "The path goes through a link (symbolic link or junction). Use the real local path.";

        /// <summary>
        /// Throws invalid_arguments (field "path") unless the file is on local storage and no component is a link. A
        /// missing file passes here; the caller reports it as document_not_found.
        /// </summary>
        public static void Require(string path, string field = "path")
        {
            try
            {
                var root = Path.GetPathRoot(path);
                if (!IsLocalRoot(root))
                {
                    throw Rejected(field);
                }

                if (!RequireLinkFreeComponents(root!, path, field))
                {
                    return;
                }

                using (var handle = CreateFileW(
                    path,
                    FileReadAttributes,
                    ShareReadWriteDelete,
                    IntPtr.Zero,
                    OpenExisting,
                    FileFlagBackupSemantics,
                    IntPtr.Zero))
                {
                    if (handle.IsInvalid)
                    {
                        var error = Marshal.GetLastWin32Error();
                        if (error == ErrorFileNotFound || error == ErrorPathNotFound)
                        {
                            return;
                        }

                        throw Rejected(field);
                    }

                    var finalPath = FinalPath(handle, field);
                    if (!finalPath.StartsWith(@"\\?\", StringComparison.Ordinal)
                        || finalPath.StartsWith(@"\\?\UNC\", StringComparison.OrdinalIgnoreCase)
                        || !IsLocalRoot(Path.GetPathRoot(finalPath.Substring(4))))
                    {
                        throw Rejected(field);
                    }
                }
            }
            catch (HostErrorException)
            {
                throw;
            }
            catch (Exception)
            {
                throw Rejected(field);
            }
        }

        /// <summary>
        /// Queries each component from the drive root down, without following it. Earlier components are verified before a
        /// later one is queried, so no unverified link is ever traversed. Returns false when a component does not exist.
        /// </summary>
        private static bool RequireLinkFreeComponents(string root, string path, string field)
        {
            var current = root;
            foreach (var component in path.Substring(root.Length).Split('\\'))
            {
                if (component.Length == 0 || component == "." || component == ".."
                    || component.IndexOfAny(ForbiddenInComponent) >= 0)
                {
                    throw Rejected(field);
                }

                current = Path.Combine(current, component);
                WIN32_FIND_DATAW data;
                var find = FindFirstFileW(current, out data);
                if (find == InvalidFindHandle)
                {
                    var error = Marshal.GetLastWin32Error();
                    if (error == ErrorFileNotFound || error == ErrorPathNotFound)
                    {
                        return false;
                    }

                    throw Rejected(field);
                }

                if (!FindClose(find))
                {
                    throw Rejected(field);
                }

                if ((data.dwFileAttributes & ReparsePointAttribute) != 0 && (data.dwReserved0 & NameSurrogateTag) != 0)
                {
                    throw HostError.InvalidArguments(field, LinkMessage);
                }
            }

            return true;
        }

        private static bool IsLocalRoot(string? root)
        {
            if (root == null || root.Length == 0 || root.StartsWith(@"\\", StringComparison.Ordinal))
            {
                return false;
            }

            var type = GetDriveTypeW(root);
            return type == DriveFixed || type == DriveRemovable || type == DriveRamDisk;
        }

        private static string FinalPath(SafeFileHandle handle, string field)
        {
            var buffer = new StringBuilder(1024);
            var length = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, VolumeNameDos);
            if (length >= buffer.Capacity)
            {
                buffer = new StringBuilder((int)length + 1);
                length = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, VolumeNameDos);
            }

            if (length == 0 || length >= buffer.Capacity)
            {
                throw Rejected(field);
            }

            return buffer.ToString();
        }

        private static HostErrorException Rejected(string field) => HostError.InvalidArguments(field, LocalOnlyMessage);

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
        private static extern uint GetDriveTypeW(string rootPathName);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateFileW(
            string fileName,
            uint desiredAccess,
            uint shareMode,
            IntPtr securityAttributes,
            uint creationDisposition,
            uint flagsAndAttributes,
            IntPtr templateFile);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern uint GetFinalPathNameByHandleW(
            SafeFileHandle file,
            StringBuilder filePath,
            uint filePathLength,
            uint flags);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr FindFirstFileW(string fileName, out WIN32_FIND_DATAW findData);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FindClose(IntPtr findFile);
    }
}
