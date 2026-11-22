using System;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// A failure with a contract error code. The request processor turns it into an error envelope.
    /// Seams throw it for domain errors (document_not_found, entity_not_found, ...).
    /// </summary>
    public sealed class HostErrorException : Exception
    {
        /// <summary>Creates the exception.</summary>
        /// <param name="code">A string from <see cref="ErrorCodes"/>.</param>
        /// <param name="message">One readable sentence, no secrets, no stack trace.</param>
        /// <param name="details">Optional details object; null becomes an empty object.</param>
        /// <param name="host">Instance facts for the envelope's host block, when an instance was involved.</param>
        public HostErrorException(string code, string message, JObject? details = null, HostInfo? host = null)
            : base(message)
        {
            Code = code;
            Details = details ?? new JObject();
            Host = host;
        }

        /// <summary>Contract error code.</summary>
        public string Code { get; }

        /// <summary>Details object placed in the error envelope.</summary>
        public JObject Details { get; }

        /// <summary>Host block for the error envelope, or null when no instance was involved.</summary>
        public HostInfo? Host { get; }
    }

    /// <summary>Factory methods for the errors Core produces itself.</summary>
    public static class HostError
    {
        /// <summary>invalid_request with a message.</summary>
        public static HostErrorException InvalidRequest(string message) =>
            new HostErrorException(ErrorCodes.InvalidRequest, message);

        /// <summary>invalid_arguments with details.field naming the offending argument.</summary>
        public static HostErrorException InvalidArguments(string field, string message) =>
            new HostErrorException(ErrorCodes.InvalidArguments, message, new JObject { ["field"] = WireJson.Value(field) });

        /// <summary>Any contract code with optional details.</summary>
        public static HostErrorException Coded(string code, string message, JObject? details = null) =>
            new HostErrorException(code, message, details);

        /// <summary>
        /// An attached SolidWorks that is running but does not answer yet (still starting, or blocked by a dialog).
        /// Code instance_timeout; the message tells the user what to do.
        /// </summary>
        public static HostErrorException AttachNotReady(string instanceId, int processId) =>
            Coded(
                ErrorCodes.InstanceTimeout,
                $"SolidWorks instance {instanceId} is running but is not answering yet. It may still be starting or waiting on a dialog; finish or dismiss the dialog in SolidWorks and try again.",
                new JObject
                {
                    ["instance_id"] = WireJson.Value(instanceId),
                    ["pid"] = WireJson.Value(processId),
                });
    }
}
