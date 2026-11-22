using System.Globalization;
using System.IO;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Locates templates for the running SolidWorks release.</summary>
    public static class ScriptTemplatePath
    {
        public static string? ForRevision(string programData, string? revision)
        {
            var year = SolidWorksVersions.RevisionToYear(revision);
            return year == null ? null : Path.Combine(programData, "SOLIDWORKS",
                "SOLIDWORKS " + year.Value.ToString(CultureInfo.InvariantCulture), "templates");
        }
    }
}
