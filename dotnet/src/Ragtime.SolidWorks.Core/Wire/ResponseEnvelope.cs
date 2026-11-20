using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>The <c>host</c> block of a response envelope. Instance fields are null when no instance was involved.</summary>
    public sealed class HostInfo
    {
        /// <summary>Creates the block.</summary>
        public HostInfo(string hostVersion, string? solidworksRevision = null, int? solidworksVersion = null, string? instanceId = null)
        {
            HostVersion = hostVersion;
            SolidworksRevision = solidworksRevision;
            SolidworksVersion = solidworksVersion;
            InstanceId = instanceId;
        }

        /// <summary>Assembly informational version of the host.</summary>
        public string HostVersion { get; }

        /// <summary>SolidWorks revision number such as "32.5.0", or null.</summary>
        public string? SolidworksRevision { get; }

        /// <summary>SolidWorks release year such as 2024, or null.</summary>
        public int? SolidworksVersion { get; }

        /// <summary>Instance id such as "i1", or null.</summary>
        public string? InstanceId { get; }

        /// <summary>Host block with no instance.</summary>
        public static HostInfo ForHost(string hostVersion) => new HostInfo(hostVersion);
    }

    /// <summary>Builds response envelopes (contract section 4). Output is always one compact line.</summary>
    public static class ResponseEnvelope
    {
        /// <summary>Success envelope with <c>result</c>.</summary>
        public static JObject Success(string? id, string? operation, JToken result, HostInfo host, long elapsedMs) =>
            new JObject
            {
                ["protocol"] = WireJson.Value(ProtocolInfo.Version),
                ["id"] = WireJson.Value(id),
                ["ok"] = WireJson.Value(true),
                ["operation"] = WireJson.Value(operation),
                ["result"] = result,
                ["host"] = ToHostBlock(host),
                ["elapsed_ms"] = WireJson.Value(elapsedMs),
            };

        /// <summary>Failure envelope with <c>error</c>.</summary>
        public static JObject Failure(
            string? id,
            string? operation,
            string code,
            string message,
            JObject? details,
            HostInfo host,
            long elapsedMs) =>
            new JObject
            {
                ["protocol"] = WireJson.Value(ProtocolInfo.Version),
                ["id"] = WireJson.Value(id),
                ["ok"] = WireJson.Value(false),
                ["operation"] = WireJson.Value(operation),
                ["error"] = new JObject
                {
                    ["code"] = WireJson.Value(code),
                    ["message"] = WireJson.Value(message),
                    ["details"] = details ?? new JObject(),
                },
                ["host"] = ToHostBlock(host),
                ["elapsed_ms"] = WireJson.Value(elapsedMs),
            };

        /// <summary>Failure envelope built from a <see cref="HostErrorException"/>; uses its host block when present.</summary>
        public static JObject FromException(string? id, string? operation, HostErrorException error, string hostVersion, long elapsedMs) =>
            Failure(id, operation, error.Code, error.Message, error.Details, error.Host ?? HostInfo.ForHost(hostVersion), elapsedMs);

        /// <summary>Serializes an envelope to one compact line without a trailing newline.</summary>
        public static string ToLine(JObject envelope) => WireJson.Compact(envelope);

        private static JObject ToHostBlock(HostInfo host) =>
            new JObject
            {
                ["host_version"] = WireJson.Value(host.HostVersion),
                ["solidworks_revision"] = WireJson.Value(host.SolidworksRevision),
                ["solidworks_version"] = WireJson.Value(host.SolidworksVersion),
                ["instance_id"] = WireJson.Value(host.InstanceId),
            };
    }

    /// <summary>Envelopes produced by the client side of <c>call</c> (no host response was received).</summary>
    public static class ClientEnvelopes
    {
        /// <summary>
        /// host_not_running with details.task_registered and details.interactive_session. Pass null when
        /// the interactive state could not be determined.
        /// </summary>
        public static JObject HostNotRunning(
            string? id,
            string? operation,
            string hostVersion,
            bool taskRegistered,
            bool? interactiveSession,
            long elapsedMs)
        {
            var details = new JObject
            {
                ["task_registered"] = WireJson.Value(taskRegistered),
                ["interactive_session"] = WireJson.Value(interactiveSession),
            };
            return ResponseEnvelope.Failure(
                id,
                operation,
                ErrorCodes.HostNotRunning,
                "The SolidWorks host is not answering. Check that the account has a desktop session and run install again.",
                details,
                HostInfo.ForHost(hostVersion),
                elapsedMs);
        }

        /// <summary>Any client-side failure with no instance involved.</summary>
        public static JObject Failure(string? id, string? operation, string hostVersion, string code, string message, long elapsedMs) =>
            ResponseEnvelope.Failure(id, operation, code, message, null, HostInfo.ForHost(hostVersion), elapsedMs);
    }
}
