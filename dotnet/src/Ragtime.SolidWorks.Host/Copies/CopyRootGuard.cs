using System.IO;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Refuses a copies tree or work folder whose ancestors are junctions or links (review finding 2), following the script-root
    /// guard: the copies root, the work folder and the application folder (RagtimeSolidWorks) are each checked. A folder that
    /// does not exist has no attributes and is not a link. Throws when attributes cannot be read; callers fail closed.
    /// </summary>
    internal static class CopyRootGuard
    {
        /// <summary>The first of the copies root, work folder and application folder that is a link, or null when none is.</summary>
        public static string? FirstLinkedCopiesAncestor(string copiesRoot)
        {
            var work = Path.GetDirectoryName(copiesRoot) ?? string.Empty;
            var application = Path.GetDirectoryName(work) ?? string.Empty;
            return ScriptFolderGuard.FirstLinkedFolder(new[] { copiesRoot, work, application }, AttributesOf);
        }

        /// <summary>The first of the work folder and the application folder that is a link, or null when neither is.</summary>
        public static string? FirstLinkedWorkFolder(string workFolder)
        {
            var application = Path.GetDirectoryName(workFolder) ?? string.Empty;
            return ScriptFolderGuard.FirstLinkedFolder(new[] { workFolder, application }, AttributesOf);
        }

        private static System.IO.FileAttributes? AttributesOf(string path) =>
            Directory.Exists(path) ? File.GetAttributes(path) : (System.IO.FileAttributes?)null;
    }
}
