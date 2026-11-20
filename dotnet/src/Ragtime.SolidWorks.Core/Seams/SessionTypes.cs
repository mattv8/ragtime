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
