using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Parses one request line into a <see cref="ParsedRequest"/>. Failures are <see cref="HostErrorException"/>s.</summary>
    public static class RequestParser
    {
        /// <summary>
        /// Parses a request line. Throws invalid_request for malformed input and protocol_mismatch when
        /// the protocol field is not 1. Never returns partially valid data.
        /// </summary>
        public static ParsedRequest Parse(string? line)
        {
            if (line == null || line.Trim().Length == 0)
            {
                throw HostError.InvalidRequest("The request is empty.");
            }

            if (WireJson.ByteCount(line) > ProtocolInfo.MaxRequestBytes)
            {
                throw HostError.InvalidRequest("The request is larger than 64 KiB.");
            }

            var root = ParseObject(line);
            CheckProtocol(root);

            return new ParsedRequest(
                RequireString(root, "id"),
                RequireString(root, "operation"),
                ReadArguments(root),
                ReadContext(root["context"]));
        }

        /// <summary>Reads a top-level string field without validating the rest; null when unreadable. Used to echo ids on errors.</summary>
        public static string? TryReadString(string? line, string name)
        {
            if (line == null || line.Trim().Length == 0)
            {
                return null;
            }

            try
            {
                var root = JToken.Parse(line) as JObject;
                if (root == null)
                {
                    return null;
                }

                var value = root[name];
                return value != null && value.Type == JTokenType.String ? (string?)value : null;
            }
            catch (JsonException)
            {
                return null;
            }
        }

        private static JObject ParseObject(string line)
        {
            JToken root;
            try
            {
                root = JToken.Parse(line);
            }
            catch (JsonException)
            {
                throw HostError.InvalidRequest("The request is not valid JSON.");
            }

            return root as JObject ?? throw HostError.InvalidRequest("The request must be a JSON object.");
        }

        private static void CheckProtocol(JObject root)
        {
            if (!WireJson.TryGetInteger(root["protocol"], out var received))
            {
                throw HostError.InvalidRequest("Field 'protocol' must be an integer.");
            }

            if (received != ProtocolInfo.Version)
            {
                throw HostError.Coded(
                    ErrorCodes.ProtocolMismatch,
                    $"This SolidWorks agent speaks protocol {ProtocolInfo.Version}, not {received}.",
                    new JObject
                    {
                        ["expected"] = WireJson.Value(ProtocolInfo.Version),
                        ["received"] = WireJson.Value(received),
                    });
            }
        }

        private static string RequireString(JObject owner, string name)
        {
            var token = owner[name];
            var value = token != null && token.Type == JTokenType.String ? (string?)token : null;
            if (value == null || value.Trim().Length == 0)
            {
                throw HostError.InvalidRequest($"Field '{name}' must be a non-empty string.");
            }

            return value;
        }

        private static JObject ReadArguments(JObject root)
        {
            var token = root["arguments"];
            if (token == null || token.Type == JTokenType.Null)
            {
                return new JObject();
            }

            return token as JObject ?? throw HostError.InvalidRequest("Field 'arguments' must be an object.");
        }

        private static RequestContext ReadContext(JToken? token)
        {
            var context = token as JObject ?? throw HostError.InvalidRequest("Field 'context' is required and must be an object.");

            int? solidworksVersion = null;
            var versionToken = context["solidworks_version"];
            if (versionToken != null && versionToken.Type != JTokenType.Null)
            {
                solidworksVersion = RequireInteger(context, "solidworks_version", ProtocolInfo.MinSolidworksYear, ProtocolInfo.MaxSolidworksYear);
            }

            var mode = RequireString(context, "instance_mode");
            if (mode != ProtocolInfo.InstanceModes.Background && mode != ProtocolInfo.InstanceModes.Attach)
            {
                throw HostError.InvalidRequest("Field 'instance_mode' must be 'background' or 'attach'.");
            }

            return new RequestContext(
                solidworksVersion,
                mode,
                RequireInteger(context, "max_instances", ProtocolInfo.MinMaxInstances, ProtocolInfo.MaxMaxInstances),
                RequireInteger(context, "idle_shutdown_seconds", ProtocolInfo.MinIdleShutdownSeconds, ProtocolInfo.MaxIdleShutdownSeconds),
                RequireBoolean(context, "allow_write"),
                RequireInteger(context, "deadline_seconds", 1, ProtocolInfo.MaxDeadlineSeconds),
                RequireInteger(context, "max_result_bytes", 1, ProtocolInfo.MaxResultBytes),
                ReadPdm(context["pdm"]),
                context.Property("allow_scripts") == null ? false : RequireBoolean(context, "allow_scripts"));
        }

        private static PdmContext? ReadPdm(JToken? token)
        {
            if (token == null || token.Type == JTokenType.Null) return null;
            var pdm = token as JObject ?? throw HostError.InvalidRequest("Field 'pdm' must be an object.");
            var vault = ReadPdmString(pdm, "vault", 1, 128);
            if (vault.IndexOfAny(new[] { '\\', '/' }) >= 0)
                throw HostError.InvalidRequest("Field 'vault' must not contain path separators.");
            var prefix = ReadPdmString(pdm, "comment_prefix", 0, 160);
            var toolId = ReadPdmString(pdm, "tool_id", 1, 64);
            if (toolId.Any(ch => !(ch >= 'A' && ch <= 'Z') && !(ch >= 'a' && ch <= 'z')
                && !(ch >= '0' && ch <= '9') && ch != '_' && ch != '-'))
                throw HostError.InvalidRequest("Field 'tool_id' must contain only letters, digits, underscores or hyphens.");
            var array = pdm["approved_folders"] as JArray;
            if (array == null || array.Count > 25)
                throw HostError.InvalidRequest("Field 'approved_folders' must be an array of at most 25 paths.");
            var folders = new List<string>();
            foreach (var item in array)
            {
                if (item.Type != JTokenType.String)
                    throw HostError.InvalidRequest("Field 'approved_folders' must contain normalized paths.");
                var folder = (string)item!;
                if (folder.Any(ch => ch < 0x20 || ch > 0x7E))
                    throw HostError.InvalidRequest("Use plain ASCII characters in approved folder paths.");
                try
                {
                    if (folder == "\\" || VaultPaths.Normalize(folder, null, "approved_folders") != folder)
                        throw HostError.InvalidRequest("Field 'approved_folders' must contain normalized non-root paths.");
                }
                catch (HostErrorException error) when (error.Code == ErrorCodes.InvalidArguments)
                {
                    throw HostError.InvalidRequest("Field 'approved_folders' must contain normalized non-root paths.");
                }
                folders.Add(folder);
            }
            return new PdmContext(vault, folders, prefix, toolId);
        }

        private static string ReadPdmString(JObject pdm, string field, int min, int max)
        {
            var token = pdm[field];
            var value = token != null && token.Type == JTokenType.String ? (string?)token : null;
            if (value == null || value.Length < min || value.Length > max)
                throw HostError.InvalidRequest($"Field '{field}' must be a string from {min} to {max} characters.");
            foreach (var ch in value)
                if (ch < 0x20 || ch == 0x7f)
                    throw HostError.InvalidRequest($"Field '{field}' must not contain control characters.");
            return value;
        }

        private static int RequireInteger(JObject owner, string name, int min, int max)
        {
            if (!WireJson.TryGetInteger(owner[name], out var value) || value < min || value > max)
            {
                throw HostError.InvalidRequest($"Field '{name}' must be an integer from {min} to {max}.");
            }

            return (int)value;
        }

        private static bool RequireBoolean(JObject owner, string name)
        {
            var token = owner[name];
            if (token == null || token.Type != JTokenType.Boolean)
            {
                throw HostError.InvalidRequest($"Field '{name}' must be a boolean.");
            }

            return (bool)token;
        }
    }
}
