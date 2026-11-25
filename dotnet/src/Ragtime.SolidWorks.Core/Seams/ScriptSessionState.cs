namespace Ragtime.SolidWorks.Core
{
    /// <summary>Application state captured before running a script and restored afterwards.</summary>
    public sealed class ScriptSessionState
    {
        /// <summary>Part document visibility, or null when unavailable.</summary>
        public bool? DocumentVisiblePart { get; set; }
        /// <summary>Assembly document visibility, or null when unavailable.</summary>
        public bool? DocumentVisibleAssembly { get; set; }
        /// <summary>Drawing document visibility, or null when unavailable.</summary>
        public bool? DocumentVisibleDrawing { get; set; }
        /// <summary>UserControl state in attach mode; null for an owned instance.</summary>
        public bool? UserControl { get; set; }
        /// <summary>Application visibility in attach mode; null for an owned instance.</summary>
        public bool? Visible { get; set; }
    }
}
