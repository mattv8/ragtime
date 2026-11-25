using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>One document known to the host. Immutable; the registry replaces entries when they change.</summary>
    public sealed class DocumentEntry
    {
        internal DocumentEntry(
            string documentId,
            string instanceId,
            string title,
            string path,
            string type,
            string? activeConfiguration,
            bool readOnly,
            bool modified,
            bool openedByHost)
        {
            DocumentId = documentId;
            InstanceId = instanceId;
            Title = title;
            Path = path;
            Type = type;
            ActiveConfiguration = activeConfiguration;
            ReadOnly = readOnly;
            Modified = modified;
            OpenedByHost = openedByHost;
        }

        /// <summary>Stable id such as "doc-1" (never reused during the host lifetime).</summary>
        public string DocumentId { get; }

        /// <summary>Instance that holds the document.</summary>
        public string InstanceId { get; }

        /// <summary>Document title.</summary>
        public string Title { get; }

        /// <summary>Full local path.</summary>
        public string Path { get; }

        /// <summary>"part", "assembly" or "drawing".</summary>
        public string Type { get; }

        /// <summary>Active configuration name, or null.</summary>
        public string? ActiveConfiguration { get; }

        /// <summary>True when read-only.</summary>
        public bool ReadOnly { get; }

        /// <summary>True when unsaved changes exist.</summary>
        public bool Modified { get; }

        /// <summary>True when the host opened the document. Only those may be closed by document.close.</summary>
        public bool OpenedByHost { get; }

        /// <summary>Wire form for documents.list and document.open.</summary>
        public DocumentItem ToItem() => new DocumentItem
        {
            DocumentId = DocumentId,
            InstanceId = InstanceId,
            Title = Title,
            Path = Path,
            Type = Type,
            ActiveConfiguration = ActiveConfiguration,
            OpenedByHost = OpenedByHost,
            ReadOnly = ReadOnly,
            Modified = Modified,
        };

        /// <summary>
        /// Reference handed to the session for this document. An unsaved document has no path: the path is null
        /// (never an empty string), so the session matches it by title.
        /// </summary>
        public DocumentRef ToRef() => new DocumentRef(string.IsNullOrEmpty(Path) ? null : Path, Title);
    }

    /// <summary>
    /// Documents known to the host: ids, instance membership and the opened-by-host flag (decision D9 and
    /// contract section 5). Ids are "doc-1", "doc-2", ... assigned in order and never reused. Thread-safe.
    /// A document is forgotten when its instance dies, when it is closed, or when the session no longer lists it.
    /// </summary>
    public sealed class DocumentRegistry
    {
        private static readonly Regex IdPattern = new Regex("^doc-[1-9][0-9]*$", RegexOptions.CultureInvariant);

        private readonly object _gate = new object();
        private readonly List<DocumentEntry> _entries = new List<DocumentEntry>();
        private int _lastNumber;

        /// <summary>True when the text has the shape of a document id (doc-N).</summary>
        public static bool LooksLikeDocumentId(string? value) => value != null && IdPattern.IsMatch(value);

        /// <summary>All documents, in id order.</summary>
        public IReadOnlyList<DocumentEntry> Snapshot()
        {
            lock (_gate)
            {
                return _entries.OrderBy(e => NumberOf(e.DocumentId)).ToList();
            }
        }

        /// <summary>Number of documents held by one instance.</summary>
        public int CountForInstance(string instanceId)
        {
            lock (_gate)
            {
                return _entries.Count(e => e.InstanceId == instanceId);
            }
        }

        /// <summary>
        /// Finds a document by id (doc-N) or, otherwise, by full path (case-insensitive). Returns null when unknown.
        /// When two instances hold the same path, the lower id wins.
        /// </summary>
        public DocumentEntry? Find(string? document) => Find(document, _ => true);

        /// <summary>
        /// Like <see cref="Find(string?)"/>, but only entries accepted by <paramref name="inScope"/> can match. Core
        /// uses it to keep a request to the documents of its own instance mode.
        /// </summary>
        public DocumentEntry? Find(string? document, Func<DocumentEntry, bool> inScope)
        {
            if (document == null)
            {
                return null;
            }

            lock (_gate)
            {
                if (LooksLikeDocumentId(document))
                {
                    return _entries.FirstOrDefault(e => e.DocumentId == document && inScope(e));
                }

                return Snapshot().FirstOrDefault(e => inScope(e) && DocumentPaths.SamePath(e.Path, document));
            }
        }

        /// <summary>
        /// Adds or updates a document reported by one instance. If the same document (instance plus path, or title
        /// when no path) is already known, its id is kept and <c>OpenedByHost</c> becomes true when
        /// <paramref name="openedByHost"/> is true. Returns the entry.
        /// </summary>
        public DocumentEntry Register(string instanceId, SessionDocumentInfo info, bool openedByHost)
        {
            lock (_gate)
            {
                var index = _entries.FindIndex(e => e.InstanceId == instanceId && SameDocument(e, info));
                if (index >= 0)
                {
                    var existing = _entries[index];
                    var updated = Create(existing.DocumentId, instanceId, info, existing.OpenedByHost || openedByHost);
                    _entries[index] = updated;
                    return updated;
                }

                var added = Create(NextId(), instanceId, info, openedByHost);
                _entries.Add(added);
                return added;
            }
        }

        /// <summary>
        /// Replaces the documents of one instance with the session's full list. Documents still listed keep their
        /// ids and flags; documents no longer listed are forgotten; new ones get new ids with OpenedByHost false.
        /// </summary>
        public void Synchronize(string instanceId, IReadOnlyList<SessionDocumentInfo> documents)
        {
            lock (_gate)
            {
                var remaining = _entries.Where(e => e.InstanceId == instanceId).ToList();
                var others = _entries.Where(e => e.InstanceId != instanceId).ToList();
                var rebuilt = new List<DocumentEntry>();

                foreach (var info in documents)
                {
                    var match = remaining.FirstOrDefault(e => SameDocument(e, info));
                    if (match != null)
                    {
                        remaining.Remove(match);
                        rebuilt.Add(Create(match.DocumentId, instanceId, info, match.OpenedByHost));
                    }
                    else
                    {
                        rebuilt.Add(Create(NextId(), instanceId, info, false));
                    }
                }

                _entries.Clear();
                _entries.AddRange(others);
                _entries.AddRange(rebuilt);
            }
        }

        /// <summary>Forgets every document of one instance (the instance died or was shut down).</summary>
        public void ForgetInstance(string instanceId)
        {
            lock (_gate)
            {
                _entries.RemoveAll(e => e.InstanceId == instanceId);
            }
        }

        /// <summary>Forgets the documents of every instance not in <paramref name="liveInstanceIds"/>.</summary>
        public void RetainInstances(IEnumerable<string> liveInstanceIds)
        {
            var live = new HashSet<string>(liveInstanceIds);
            lock (_gate)
            {
                _entries.RemoveAll(e => !live.Contains(e.InstanceId));
            }
        }

        /// <summary>Forgets one document (after document.close). Returns false when the id is unknown.</summary>
        public bool Remove(string documentId)
        {
            lock (_gate)
            {
                return _entries.RemoveAll(e => e.DocumentId == documentId) > 0;
            }
        }

        private static bool SameDocument(DocumentEntry entry, SessionDocumentInfo info) =>
            SameDocument(entry.Path, entry.Title, info.Path, info.Title);

        private static bool SameDocument(string entryPath, string entryTitle, string infoPath, string infoTitle)
        {
            if (!string.IsNullOrEmpty(entryPath) && !string.IsNullOrEmpty(infoPath))
            {
                return DocumentPaths.SamePath(entryPath, infoPath);
            }

            return string.Equals(entryTitle, infoTitle, StringComparison.OrdinalIgnoreCase);
        }

        private static DocumentEntry Create(string documentId, string instanceId, SessionDocumentInfo info, bool openedByHost) =>
            new DocumentEntry(
                documentId,
                instanceId,
                info.Title,
                info.Path,
                info.Type,
                info.ActiveConfiguration,
                info.ReadOnly,
                info.Modified,
                openedByHost);

        private string NextId()
        {
            _lastNumber++;
            return "doc-" + _lastNumber.ToString(CultureInfo.InvariantCulture);
        }

        private static int NumberOf(string documentId) =>
            int.Parse(documentId.Substring("doc-".Length), CultureInfo.InvariantCulture);
    }
}
