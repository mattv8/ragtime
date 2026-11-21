using System;
using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Per-instance session: all SolidWorks reads for one running instance. Implemented by the Readers lane
    /// (Ragtime.SolidWorks.Readers) on top of the typed interop API.
    ///
    /// Threading: every member is called synchronously on that instance's STA worker thread, one call at a time.
    /// Members must not block on other instances and must not start threads.
    ///
    /// Errors: throw <see cref="HostErrorException"/> for domain errors:
    /// document_not_found (path missing on disk when opening), document_not_open (path not open in this instance),
    /// entity_not_found (named sketch, feature or dimension does not exist), not_supported_for_document_type,
    /// solidworks_error (SolidWorks reported a failure; put the API message in the exception message).
    /// Other exceptions are reported as internal_error, so translate COM exceptions into solidworks_error.
    ///
    /// Lists: list methods return the FULL list. Core applies paging, the byte budget and snapshots.
    ///
    /// Units: all quantities are SI (metres, kilograms, seconds, radians). Never return inches or degrees.
    ///
    /// Read-only: no member may modify, save, rebuild or close a model. The only state change is
    /// <see cref="OpenReadOnlySilent"/> and <see cref="CloseDocument"/>, which never saves.
    /// </summary>
    public interface ISolidWorksSession
    {
        /// <summary>SLDWORKS process id of this session's instance.</summary>
        int ProcessId { get; }
        /// <summary>Start time (UTC) of that process, for pid-reuse checks.</summary>
        DateTime ProcessStartUtc { get; }
        /// <summary>True in attach mode (a user's interactive SolidWorks).</summary>
        bool IsAttached { get; }
        /// <summary>Re-applies the host's window hiding to an owned instance after a script; no-op when attached.</summary>
        void ReapplyWindowHiding();

        /// <summary>Captures application state before a script, on this instance's STA thread.</summary>
        ScriptSessionState CaptureScriptState();

        /// <summary>Restores captured application state on this instance's STA thread; COM errors are handled best effort and never thrown.</summary>
        void RestoreScriptState(ScriptSessionState state);

        /// <summary>Revision, process id and visibility of the SolidWorks application behind this session.</summary>
        InstanceFacts GetInstanceFacts();

        /// <summary>All documents open in this instance, including those the host did not open.</summary>
        IReadOnlyList<SessionDocumentInfo> ListOpenDocuments();

        /// <summary>
        /// Opens a file read-only and silently (no dialogs, no save, no rebuild). If the document is already open
        /// in this instance, returns its info with <see cref="SessionDocumentInfo.WasAlreadyOpen"/> = true.
        /// <paramref name="configuration"/> null means the file's active configuration.
        /// Throws document_not_found when the file does not exist and solidworks_error when SolidWorks cannot open it.
        /// </summary>
        SessionDocumentInfo OpenReadOnlySilent(string path, string? configuration);

        /// <summary>Closes a document without saving. Only called for documents the host opened.</summary>
        void CloseDocument(DocumentRef document);

        /// <summary>Summary of a model (type-specific counts). <see cref="ModelSummary.DocumentId"/> may be left empty.</summary>
        ModelSummary GetModelSummary(DocumentRef document);

        /// <summary>All configurations of the model, in SolidWorks order.</summary>
        IReadOnlyList<ConfigurationItem> ListConfigurations(DocumentRef document);

        /// <summary>
        /// Feature tree in tree order. When <paramref name="includeSystem"/> is false, folders such as Comments and
        /// Sensors are hidden. When <paramref name="types"/> is not null, only features whose type is in the list.
        /// </summary>
        IReadOnlyList<FeatureItem> ListFeatures(DocumentRef document, bool includeSystem, IReadOnlyList<string>? types);

        /// <summary>One feature with its parents, children, dimensions and sketch. Throws entity_not_found for an unknown name.</summary>
        FeatureDetail GetFeature(DocumentRef document, string feature);

        /// <summary>Counts and constraint status of a sketch. Throws entity_not_found for an unknown sketch name.</summary>
        SketchSummary GetSketchSummary(DocumentRef document, string sketch);

        /// <summary>Segments of a sketch in sketch coordinates (metres).</summary>
        IReadOnlyList<SegmentItem> ListSketchSegments(DocumentRef document, string sketch);

        /// <summary>Points of a sketch.</summary>
        IReadOnlyList<PointItem> ListSketchPoints(DocumentRef document, string sketch);

        /// <summary>Geometric relations of a sketch.</summary>
        IReadOnlyList<RelationItem> ListSketchRelations(DocumentRef document, string sketch);

        /// <summary>Dimensions that belong to a sketch.</summary>
        IReadOnlyList<DimensionItem> ListSketchDimensions(DocumentRef document, string sketch);

        /// <summary>All dimensions, or only those of <paramref name="feature"/> when it is not null.</summary>
        IReadOnlyList<DimensionItem> ListDimensions(DocumentRef document, string? feature);

        /// <summary>Mates of an assembly. Core rejects non-assemblies before calling.</summary>
        IReadOnlyList<MateItem> ListMates(DocumentRef document);

        /// <summary>Components of an assembly; top level only when <paramref name="topLevelOnly"/> is true.</summary>
        IReadOnlyList<ComponentItem> ListComponents(DocumentRef document, bool topLevelOnly);

        /// <summary>File-level properties, or the properties of <paramref name="configuration"/> when it is not null.</summary>
        PropertiesResult GetProperties(DocumentRef document, string? configuration);

        /// <summary>Mass properties in SI units. Throws solidworks_error when the model has no solid body.</summary>
        MassPropertiesResult GetMassProperties(DocumentRef document);

        /// <summary>All equations of the model.</summary>
        IReadOnlyList<EquationItem> ListEquations(DocumentRef document);
    }

    /// <summary>
    /// Creates a session from a SolidWorks application object. Implemented by the Readers lane and called by the
    /// instance manager on the instance's STA thread, right after the application object is obtained.
    /// </summary>
    public interface ISolidWorksSessionFactory
    {
        /// <summary>
        /// Wraps the application object (the SolidWorks COM application, typed as the interop <c>ISldWorks</c>
        /// behind <c>object</c>). Called on the instance's STA thread. Must not throw for a healthy application.
        /// <paramref name="options"/> tells the session whether the instance is attached and how to re-hide it.
        /// </summary>
        ISolidWorksSession Create(object solidworksApplication, SessionOptions options);
    }
}
