using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Exact error code strings of contract section 4. Never rename them.</summary>
    public static class ErrorCodes
    {
        /// <summary>Request line is not a valid v1 request.</summary>
        public const string InvalidRequest = "invalid_request";

        /// <summary>Request protocol is not version 1.</summary>
        public const string ProtocolMismatch = "protocol_mismatch";

        /// <summary>Operation name is not in the registry.</summary>
        public const string UnknownOperation = "unknown_operation";

        /// <summary>Arguments do not match the operation schema, or a path/option is refused.</summary>
        public const string InvalidArguments = "invalid_arguments";

        /// <summary>Operation needs allow_write and the context does not allow it.</summary>
        public const string WriteNotAllowed = "write_not_allowed";

        /// <summary>The client could not reach a running host.</summary>
        public const string HostNotRunning = "host_not_running";

        /// <summary>The host is not running in an interactive desktop session.</summary>
        public const string NoInteractiveSession = "no_interactive_session";

        /// <summary>The pinned SolidWorks year is not installed.</summary>
        public const string VersionNotInstalled = "version_not_installed";

        /// <summary>The attached instance is not the pinned year.</summary>
        public const string VersionMismatch = "version_mismatch";

        /// <summary>SolidWorks could not be started.</summary>
        public const string SolidworksStartFailed = "solidworks_start_failed";

        /// <summary>Attach mode and no SolidWorks instance in this session.</summary>
        public const string SolidworksNotRunning = "solidworks_not_running";

        /// <summary>Every allowed instance is busy and no new one may start.</summary>
        public const string InstanceLimit = "instance_limit";

        /// <summary>An instance did not become ready, or an attached instance is stuck.</summary>
        public const string InstanceTimeout = "instance_timeout";

        /// <summary>The call passed its deadline.</summary>
        public const string OperationTimeout = "operation_timeout";

        /// <summary>No document with that id or path is known.</summary>
        public const string DocumentNotFound = "document_not_found";

        /// <summary>The document path is not open.</summary>
        public const string DocumentNotOpen = "document_not_open";

        /// <summary>The operation does not apply to this document type.</summary>
        public const string NotSupportedForDocumentType = "not_supported_for_document_type";

        /// <summary>The named sketch, feature or entity does not exist.</summary>
        public const string EntityNotFound = "entity_not_found";

        /// <summary>The list changed since the snapshot was taken.</summary>
        public const string StaleSnapshot = "stale_snapshot";

        /// <summary>The result, or one item of it, exceeds max_result_bytes.</summary>
        public const string ResultTooLarge = "result_too_large";

        /// <summary>SolidWorks reported an error.</summary>
        public const string SolidworksError = "solidworks_error";

        /// <summary>Unexpected failure inside the host.</summary>
        public const string InternalError = "internal_error";

        public const string PdmNotLinked = "pdm_not_linked";
        public const string PdmUnavailable = "pdm_unavailable";
        public const string PdmFolderNotApproved = "pdm_folder_not_approved";
        public const string PdmNotFound = "pdm_not_found";
        public const string PdmConflict = "pdm_conflict";
        public const string PdmError = "pdm_error";

        /// <summary>Every code Core may produce, in contract order.</summary>
        public static IReadOnlyList<string> All { get; } = new[]
        {
            InvalidRequest, ProtocolMismatch, UnknownOperation, InvalidArguments, WriteNotAllowed,
            HostNotRunning, NoInteractiveSession, VersionNotInstalled, VersionMismatch,
            SolidworksStartFailed, SolidworksNotRunning, InstanceLimit, InstanceTimeout,
            OperationTimeout, DocumentNotFound, DocumentNotOpen, NotSupportedForDocumentType,
            EntityNotFound, StaleSnapshot, ResultTooLarge, SolidworksError, InternalError,
            PdmNotLinked, PdmUnavailable, PdmFolderNotApproved, PdmNotFound, PdmConflict, PdmError,
        };
    }
}
