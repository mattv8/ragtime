using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>One file of the <c>files</c> list of <c>document.copy</c>.</summary>
    public sealed class CopyFileItem
    {
        /// <summary>Path of the file the document or reference came from. Never changed by the copy.</summary>
        public string Source { get; set; } = string.Empty;

        /// <summary>Path of the copy inside the copy folder.</summary>
        public string Path { get; set; } = string.Empty;
    }

    /// <summary>Result of <c>document.copy</c>. Files are copied into <see cref="Folder"/>; the source is not touched.</summary>
    public sealed class DocumentCopyResult
    {
        /// <summary>The copy folder, <c>&lt;work folder&gt;\copies\&lt;id&gt;\</c>.</summary>
        public string Folder { get; set; } = string.Empty;

        /// <summary>Path of the copied top document.</summary>
        public string Path { get; set; } = string.Empty;

        /// <summary>The first copied files, the top document first; at most 50 entries so the result fits the budget.</summary>
        public List<CopyFileItem> Files { get; set; } = new List<CopyFileItem>();

        /// <summary>Every copied file, including those not listed in <see cref="Files"/>.</summary>
        public int Count { get; set; }

        /// <summary>True when <see cref="Count"/> is more than the entries listed in <see cref="Files"/>. Always serialized.</summary>
        public bool FilesTruncated { get; set; }

        /// <summary>Notes for the chat agent: missing source files and references Pack and Go left outside the copy.</summary>
        public List<string> Warnings { get; set; } = new List<string>();
    }
}
