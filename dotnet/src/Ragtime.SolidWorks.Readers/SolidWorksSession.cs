using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Read-only view of one SolidWorks instance through the typed interop API. Every member runs on the instance's
    /// STA thread, one call at a time, and never saves, rebuilds, activates a configuration or modifies a model. COM
    /// failures become solidworks_error. This file holds the shell (error translation and document lookup); the partial
    /// files hold the documents, features, sketches, dimensions, assemblies and properties readers. The only cached
    /// object is the application itself.
    /// </summary>
    internal sealed partial class SolidWorksSession : ISolidWorksSession
    {
        private readonly SW.ISldWorks _application;

        internal SolidWorksSession(SW.ISldWorks application)
        {
            _application = application;
        }

        /// <summary>Revision, process id and visibility of the application.</summary>
        public InstanceFacts GetInstanceFacts() => Run("reading the SolidWorks instance", () => new InstanceFacts(
            ComHelper.TryText(() => _application.RevisionNumber()),
            _application.GetProcessID(),
            _application.Visible));

        /// <summary>Runs one read, translating COM, cast and null failures into solidworks_error.</summary>
        private static T Run<T>(string action, Func<T> work)
        {
            try
            {
                return work();
            }
            catch (HostErrorException)
            {
                throw;
            }
            catch (COMException exception)
            {
                throw ComHelper.Translate(action, exception);
            }
            catch (InvalidCastException exception)
            {
                throw ComHelper.Translate(action, exception);
            }
            catch (NullReferenceException exception)
            {
                throw ComHelper.Translate(action, exception);
            }
        }

        /// <summary>Runs one read that returns nothing, with the same error translation as <see cref="Run{T}"/>.</summary>
        private static void RunVoid(string action, Action work)
        {
            Run(action, () =>
            {
                work();
                return true;
            });
        }

        /// <summary>All documents open in the application, in SolidWorks order. Repeats and overflow fail.</summary>
        private List<SW.ModelDoc2> OpenDocuments() => ComWalk.Chain<SW.ModelDoc2>(
            "The open document list",
            ComWalk.DocumentLimit,
            () => _application.GetFirstDocument() as SW.ModelDoc2,
            document => document.GetNext() as SW.ModelDoc2);

        /// <summary>
        /// Finds an open document: by path (case-insensitive) when the reference has one, otherwise by title.
        /// Returns null when none matches.
        /// </summary>
        private SW.ModelDoc2? FindOpenDocument(DocumentRef reference)
        {
            foreach (var document in OpenDocuments())
            {
                if (Matches(document, reference))
                {
                    return document;
                }
            }

            return null;
        }

        /// <summary>The open document for a reference, or document_not_open.</summary>
        private SW.ModelDoc2 RequireOpenDocument(DocumentRef reference) =>
            FindOpenDocument(reference)
            ?? throw HostError.Coded(ErrorCodes.DocumentNotOpen, "The document is not open in this SolidWorks instance.");

        /// <summary>
        /// True when the document matches the reference. A path is used only when it is not empty; an empty or null path
        /// means match by title, so unsaved documents never match each other by an empty path.
        /// </summary>
        private static bool Matches(SW.ModelDoc2 document, DocumentRef reference)
        {
            if (!string.IsNullOrEmpty(reference.Path))
            {
                return DocumentPaths.SamePath(ComHelper.TryText(() => document.GetPathName()), reference.Path);
            }

            return !string.IsNullOrEmpty(reference.Title)
                && string.Equals(ComHelper.TryText(() => document.GetTitle()), reference.Title, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>The active configuration name of a document, or null when unavailable.</summary>
        private static string? ActiveConfigurationName(SW.ModelDoc2 document) =>
            ComHelper.TryText(() => ((SW.Configuration)document.GetActiveConfiguration())?.Name);
    }
}
