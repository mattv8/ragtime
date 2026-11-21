using System;
using System.Collections.Generic;
using System.IO;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Pure rule for the script folders the host must not write or delete through a junction or link.</summary>
    public static class ScriptFolderGuard
    {
        /// <summary>
        /// Returns the first folder, in the given order, that is a reparse point, or null when none is. A folder whose
        /// attributes are unknown (null: it does not exist) is not a link. The lookup may throw; callers fail closed.
        /// </summary>
        public static string? FirstLinkedFolder(IEnumerable<string> folders, Func<string, FileAttributes?> attributesOf)
        {
            foreach (var folder in folders)
            {
                var attributes = attributesOf(folder);
                if (attributes.HasValue && ScriptRunEntryRule.IsLink(attributes.Value))
                {
                    return folder;
                }
            }

            return null;
        }
    }
}
