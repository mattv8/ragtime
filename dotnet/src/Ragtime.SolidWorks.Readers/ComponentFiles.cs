using System;
using System.IO;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Whether a component's or reference's file is on local disk, and how a component's suppression state is named.
    /// Network (UNC, device) and other non-local paths are not checked: they count as not existing and cause no I/O.
    /// Stateless.
    /// </summary>
    internal static class ComponentFiles
    {
        /// <summary>
        /// True when <paramref name="path"/> names a local file that exists. Empty, invalid and non-local paths give false
        /// before any file-system call. A link on the path is rejected by <see cref="LocalFileCheck"/> (false).
        /// </summary>
        public static bool LocalFileExists(string? path)
        {
            if (DocumentPaths.FindLocalFileProblem(path) != null)
            {
                return false;
            }

            try
            {
                LocalFileCheck.Require(path!, "path");
            }
            catch (HostErrorException)
            {
                return false;
            }

            return File.Exists(path);
        }

        /// <summary>
        /// The protocol name of a GetSuppression2 state. Only read for a component whose file exists; a missing file is
        /// reported as missing_file. swComponentInternalIdMismatch and the resolved states give resolved.
        /// </summary>
        public static string SuppressionOf(int state)
        {
            switch ((SWC.swComponentSuppressionState_e)state)
            {
                case SWC.swComponentSuppressionState_e.swComponentSuppressed:
                    return ComponentSuppression.Suppressed;
                case SWC.swComponentSuppressionState_e.swComponentLightweight:
                case SWC.swComponentSuppressionState_e.swComponentFullyLightweight:
                    return ComponentSuppression.Lightweight;
                default:
                    return ComponentSuppression.Resolved;
            }
        }

        /// <summary>True when the component is loaded in memory or its referenced file exists.</summary>
        public static bool IsResolved(SW.Component2 component)
        {
            if (ComHelper.TryObject(() => component.GetModelDoc2()) != null)
            {
                return true;
            }

            return LocalFileExists(ComHelper.TryText(() => component.GetPathName()));
        }
    }
}
