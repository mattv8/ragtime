using System;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Hints for syntax unsupported by the Windows C# 5 compiler.</summary>
    public static class ScriptCompileHints
    {
        /// <summary>Returns the guidance hint for known newer syntax errors.</summary>
        public static string? HintFor(string code, string message)
        {
            if (code == "CS1056" || code == "CS1525" || code == "CS1003"
                || (code == "CS0103" && message.IndexOf("'nameof'", StringComparison.Ordinal) >= 0))
            {
                return "C# 5 only; see the scripts guidance.";
            }
            return null;
        }
    }
}
