using System;
using System.IO;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// Applies the retention rule to script run folders. Junctions and links are never followed: an expired one is unlinked.
    /// Kept runs are for review.
    /// </summary>
    internal static class ScriptRunSweeper
    {
        /// <summary>Applies the retention rule under <paramref name="root"/>. Never throws; failures are logged.</summary>
        public static void Sweep(string root, DateTime nowUtc, Action<string> log)
        {
            try
            {
                if (!Directory.Exists(root))
                {
                    return;
                }

                var linked = ScriptRootGuard.FirstLinkedFolder(root);
                if (linked != null)
                {
                    log("script run sweep skipped: " + linked + " is a junction or link.");
                    return;
                }

                foreach (var folder in Directory.GetDirectories(root))
                {
                    SweepEntry(folder, nowUtc, log);
                }
            }
            catch (Exception exception)
            {
                log("script run sweep failed: " + exception.Message);
            }
        }

        private static void SweepEntry(string entry, DateTime nowUtc, Action<string> log)
        {
            try
            {
                var action = ScriptRunEntryRule.Decide(File.GetAttributes(entry), Directory.GetCreationTimeUtc(entry), nowUtc);
                switch (action)
                {
                    case ScriptRunEntryAction.Delete:
                        Directory.Delete(entry, true);
                        break;
                    case ScriptRunEntryAction.UnlinkOnly:
                        Directory.Delete(entry, false);
                        break;
                }
            }
            catch (Exception exception)
            {
                log("script run sweep could not remove " + Path.GetFileName(entry) + ": " + exception.Message);
            }
        }
    }
}
