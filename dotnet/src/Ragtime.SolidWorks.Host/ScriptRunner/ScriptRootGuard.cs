using System;
using System.IO;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Refuses script folders that are junctions or links, so a run never writes or deletes outside its root. The scripts
    /// root and its parent application folder (<c>RagtimeSolidWorks</c>) are both checked.
    /// </summary>
    internal static class ScriptRootGuard
    {
        /// <summary>
        /// The first of the application folder and the scripts root that is a junction or link, or null when neither is.
        /// Throws when attributes cannot be read; callers fail closed.
        /// </summary>
        public static string? FirstLinkedFolder(string scriptsRoot)
        {
            var applicationFolder = Path.GetDirectoryName(scriptsRoot) ?? string.Empty;
            return ScriptFolderGuard.FirstLinkedFolder(new[] { applicationFolder, scriptsRoot }, AttributesOf);
        }

        /// <summary>Throws when a single folder, such as a new run folder, is a junction or link.</summary>
        public static void ThrowIfLinked(string folder)
        {
            if (ScriptFolderGuard.FirstLinkedFolder(new[] { folder }, AttributesOf) != null)
            {
                throw new IOException("Refusing a junction or link as a script run folder: " + folder);
            }
        }

        private static FileAttributes? AttributesOf(string path) =>
            Directory.Exists(path) ? File.GetAttributes(path) : (FileAttributes?)null;
    }
}
