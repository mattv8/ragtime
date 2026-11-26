using System;
using System.Collections.Generic;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Runs one copy retention sweep over the disk and logs its counts. Nothing is enumerated or deleted unless the copies
    /// root's ancestors are free of links (review finding 2). The open documents come from the instance manager; when they
    /// cannot be listed the sweep is skipped, and an unreadable document path makes Core delete nothing (review finding 3).
    /// Never throws. Logs carry counts and exception type names only, never file paths.
    /// </summary>
    internal static class CopySweeper
    {
        /// <summary>Sweeps expired copies under <paramref name="copiesRoot"/>, keeping folders with a document in the open paths.</summary>
        public static void Sweep(string copiesRoot, Func<IReadOnlyList<string?>?> openPaths, DateTime nowUtc, Action<string> log)
        {
            try
            {
                var linked = CopyRootGuard.FirstLinkedCopiesAncestor(copiesRoot);
                if (linked != null)
                {
                    log("copy retention skipped: the copies folder or a folder above it is a junction or link.");
                    return;
                }

                var open = openPaths();
                if (open == null)
                {
                    log("copy retention skipped: an open SolidWorks instance did not answer.");
                    return;
                }

                var result = new CopyRetention(new DiskCopyFileSystem()).Sweep(copiesRoot, nowUtc, open);
                if (!result.OpenStateKnown)
                {
                    log("copy retention skipped: the path of an open document could not be read.");
                    return;
                }

                log("copy retention: removed " + result.DeletedFolders + " folder(s), kept " + result.SkippedOpenFolders
                    + " open, " + result.FailedPaths.Count + " item(s) to retry.");
            }
            catch (Exception exception)
            {
                log("copy retention failed: " + exception.GetType().Name);
            }
        }
    }
}
