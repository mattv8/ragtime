using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// The disk checks of <c>document.copy</c> that run before any save (review findings 1 and 7): reparse points on the copy
    /// folder and its ancestors, and the size budget. The decisions are Core's (<see cref="CopyDestination"/>,
    /// <see cref="CopyQuota"/>); this class only reads the disk. Stateless.
    /// </summary>
    internal static class CopyGuards
    {
        /// <summary>
        /// Refuses a copy folder when the folder itself, the copies folder, the work folder or the application folder
        /// (RagtimeSolidWorks) is a junction or link. A folder that does not exist yet has no attributes and is not a link.
        /// </summary>
        public static void RequirePlainChain(string folder)
        {
            var copies = Path.GetDirectoryName(folder) ?? string.Empty;
            var work = Path.GetDirectoryName(copies) ?? string.Empty;
            var application = Path.GetDirectoryName(work) ?? string.Empty;
            var linked = ScriptFolderGuard.FirstLinkedFolder(new[] { folder, copies, work, application }, AttributesOf);
            if (linked != null)
            {
                throw HostError.InvalidArguments("folder", "The copy folder or a folder above it is a junction or link. Nothing was copied.");
            }
        }

        /// <summary>
        /// Refuses the copy before it is written when its source files exceed the size limits or the copies folder and the
        /// volume are over their quota (<see cref="CopyQuota.RequireWithinLimits"/>). Sizes are summed without overflow;
        /// a missing source counts as zero bytes.
        /// </summary>
        public static void RequireBudget(IEnumerable<string> sources, string folder)
        {
            var copyBytes = CopyQuota.SumSizes(sources.Select(SizeOf));
            var existingBytes = PlainTreeBytes(Path.GetDirectoryName(folder) ?? string.Empty);
            var freeBytes = new DriveInfo(Path.GetPathRoot(folder)).AvailableFreeSpace;
            CopyQuota.RequireWithinLimits(copyBytes, existingBytes, freeBytes);
        }

        private static long SizeOf(string path) => File.Exists(path) ? new FileInfo(path).Length : 0L;

        /// <summary>Total size of the files under a folder. Junctions and links below it are not followed or counted.</summary>
        private static long PlainTreeBytes(string folder)
        {
            if (!Directory.Exists(folder) || IsLink(folder))
            {
                return 0L;
            }

            var total = 0L;
            foreach (var file in Directory.GetFiles(folder))
            {
                total = CopyQuota.Add(total, new FileInfo(file).Length);
            }

            foreach (var subfolder in Directory.GetDirectories(folder))
            {
                total = CopyQuota.Add(total, PlainTreeBytes(subfolder));
            }

            return total;
        }

        private static FileAttributes? AttributesOf(string path) =>
            Directory.Exists(path) ? File.GetAttributes(path) : (FileAttributes?)null;

        private static bool IsLink(string path) => (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0;
    }
}
