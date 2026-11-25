using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Identifies an open document for the per-instance session. Core fills it from its registry; the
    /// session may match on <see cref="Path"/> (preferred) or <see cref="Title"/>. At least one is set.
    /// </summary>
    public sealed class DocumentRef
    {
        /// <summary>Creates a reference.</summary>
        public DocumentRef(string? path, string? title)
        {
            Path = path;
            Title = title;
        }

        /// <summary>Full local path, or null.</summary>
        public string? Path { get; }

        /// <summary>Document title, or null.</summary>
        public string? Title { get; }
    }

    /// <summary>
    /// Facts about an open document as the session reports them. <see cref="WasAlreadyOpen"/> is true when
    /// <c>OpenReadOnlySilent</c> found the document already open (then the host did not open it and must not close it).
    /// </summary>
    public sealed class SessionDocumentInfo
    {
        /// <summary>Creates the info.</summary>
        public SessionDocumentInfo(
            string title,
            string path,
            string type,
            string? activeConfiguration,
            bool readOnly,
            bool modified,
            bool wasAlreadyOpen = false)
        {
            Title = title;
            Path = path;
            Type = type;
            ActiveConfiguration = activeConfiguration;
            ReadOnly = readOnly;
            Modified = modified;
            WasAlreadyOpen = wasAlreadyOpen;
        }

        /// <summary>Document title.</summary>
        public string Title { get; }

        /// <summary>Full local path. Must be set; Core matches documents by it.</summary>
        public string Path { get; }

        /// <summary>"part", "assembly" or "drawing".</summary>
        public string Type { get; }

        /// <summary>Active configuration name, or null.</summary>
        public string? ActiveConfiguration { get; }

        /// <summary>True when open read-only.</summary>
        public bool ReadOnly { get; }

        /// <summary>True when unsaved changes exist.</summary>
        public bool Modified { get; }

        /// <summary>True when the document was already open before <c>OpenReadOnlySilent</c> was called.</summary>
        public bool WasAlreadyOpen { get; }
    }

    /// <summary>
    /// Asks the session to write one export file. <see cref="TargetPath"/> is chosen by Core (a generated name inside the
    /// work folder); the session writes exactly that path and nothing else.
    /// </summary>
    public sealed class ExportRequest
    {
        /// <summary>Creates the request.</summary>
        public ExportRequest(DocumentRef document, string format, string? configuration, string targetPath)
        {
            Document = document;
            Format = format;
            Configuration = configuration;
            TargetPath = targetPath;
        }

        /// <summary>The open document to export.</summary>
        public DocumentRef Document { get; }

        /// <summary>step, stl, 3mf, glb or native (validated by Core).</summary>
        public string Format { get; }

        /// <summary>Configuration to export, or null for the active one.</summary>
        public string? Configuration { get; }

        /// <summary>Full path of the file to write.</summary>
        public string TargetPath { get; }
    }

    /// <summary>
    /// Asks the session to render one PNG of a document's named view. <see cref="TargetPath"/> is chosen by Core.
    /// <see cref="Width"/> and <see cref="Height"/> are already clamped by Core.
    /// </summary>
    public sealed class PreviewRequest
    {
        /// <summary>Creates the request.</summary>
        public PreviewRequest(DocumentRef document, string view, int width, int height, string targetPath, string? configuration)
        {
            Document = document;
            View = view;
            Width = width;
            Height = height;
            TargetPath = targetPath;
            Configuration = configuration;
        }

        /// <summary>Configuration to show for the picture, or null for the active one (the one the export used).</summary>
        public string? Configuration { get; }

        /// <summary>The open document to render.</summary>
        public DocumentRef Document { get; }

        /// <summary>isometric, front, top or right.</summary>
        public string View { get; }

        /// <summary>Width in pixels.</summary>
        public int Width { get; }

        /// <summary>Height in pixels.</summary>
        public int Height { get; }

        /// <summary>Full path of the PNG to write.</summary>
        public string TargetPath { get; }
    }

    /// <summary>One file the session wrote.</summary>
    public sealed class ExportedFile
    {
        /// <summary>Creates the entry.</summary>
        public ExportedFile(string role, string path, string format)
        {
            Role = role;
            Path = path;
            Format = format;
        }

        /// <summary>model, native or preview.</summary>
        public string Role { get; }

        /// <summary>The path that was written (the request's target path).</summary>
        public string Path { get; }

        /// <summary>step, stl, 3mf, glb, png, sldprt, sldasm, slddrw or zip.</summary>
        public string Format { get; }
    }

    /// <summary>
    /// What the session wrote: the files, the configuration that was exported and any warnings (for example, that
    /// dependencies were missing or the document had unsaved changes). Core adds the file names, sizes and hashes.
    /// </summary>
    public sealed class ExportResult
    {
        /// <summary>Creates the result.</summary>
        public ExportResult(IReadOnlyList<ExportedFile> files, string? configuration, IReadOnlyList<string> warnings)
        {
            Files = files;
            Configuration = configuration;
            Warnings = warnings;
        }

        /// <summary>Files written, in order.</summary>
        public IReadOnlyList<ExportedFile> Files { get; }

        /// <summary>Configuration exported, or null when unknown.</summary>
        public string? Configuration { get; }

        /// <summary>Human-readable notes for the chat agent.</summary>
        public IReadOnlyList<string> Warnings { get; }
    }

    /// <summary>Instance facts returned by <c>ISolidWorksSession.GetInstanceFacts</c>.</summary>
    public sealed class InstanceFacts
    {
        /// <summary>Creates the facts.</summary>
        public InstanceFacts(string? revision, int processId, bool visible)
        {
            Revision = revision;
            ProcessId = processId;
            Visible = visible;
        }

        /// <summary>Revision number such as "32.5.0" (SolidWorks RevisionNumber), or null if unreadable.</summary>
        public string? Revision { get; }

        /// <summary>Process id of the SLDWORKS process.</summary>
        public int ProcessId { get; }

        /// <summary>Visible state of the application window.</summary>
        public bool Visible { get; }
    }
}
