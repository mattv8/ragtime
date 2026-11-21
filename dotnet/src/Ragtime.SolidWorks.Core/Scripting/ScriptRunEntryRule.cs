using System;
using System.IO;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>What the sweeper does with one child of the scripts folder.</summary>
    public enum ScriptRunEntryAction
    {
        /// <summary>Not expired: leave it alone.</summary>
        Keep,

        /// <summary>Expired real folder: delete it with its contents.</summary>
        Delete,

        /// <summary>Expired junction or symbolic link: remove the link only, never its target or contents.</summary>
        UnlinkOnly,
    }

    /// <summary>Decides from attributes whether a run folder entry is kept, deleted, or unlinked without following it.</summary>
    public static class ScriptRunEntryRule
    {
        /// <summary>True for a junction, symbolic link or other reparse point.</summary>
        public static bool IsLink(FileAttributes attributes) => (attributes & FileAttributes.ReparsePoint) != 0;

        /// <summary>
        /// Keeps entries younger than <see cref="ScriptRunRetention.MaxAge"/>. An expired reparse point is unlinked, never
        /// followed; an expired real folder is deleted.
        /// </summary>
        public static ScriptRunEntryAction Decide(FileAttributes attributes, DateTime createdUtc, DateTime nowUtc)
        {
            if (!ScriptRunRetention.IsExpired(createdUtc, nowUtc))
            {
                return ScriptRunEntryAction.Keep;
            }

            return IsLink(attributes) ? ScriptRunEntryAction.UnlinkOnly : ScriptRunEntryAction.Delete;
        }
    }
}
