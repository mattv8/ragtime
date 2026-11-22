using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Paging arguments of a list operation: offset, limit (1-500, default 100) and optional snapshot.</summary>
    public sealed class PageRequest
    {
        private static readonly Regex SnapshotPattern = new Regex("^[0-9a-f]{64}$", RegexOptions.CultureInvariant);

        /// <summary>Creates a page request.</summary>
        public PageRequest(int offset, int limit, string? snapshot)
        {
            Offset = offset;
            Limit = limit;
            Snapshot = snapshot;
        }

        /// <summary>Index of the first item, 0-based.</summary>
        public int Offset { get; }

        /// <summary>Maximum number of items in the page.</summary>
        public int Limit { get; }

        /// <summary>Snapshot the caller saw on an earlier page, or null.</summary>
        public string? Snapshot { get; }

        /// <summary>Reads offset, limit and snapshot from an arguments object; throws invalid_arguments when malformed.</summary>
        public static PageRequest FromArguments(JObject arguments)
        {
            var offset = ReadInteger(arguments, "offset", 0, int.MaxValue, 0);
            var limit = ReadInteger(arguments, "limit", ProtocolInfo.MinPageLimit, ProtocolInfo.MaxPageLimit, ProtocolInfo.DefaultPageLimit);
            return new PageRequest(offset, limit, ReadSnapshot(arguments));
        }

        private static int ReadInteger(JObject arguments, string name, int min, int max, int fallback)
        {
            var token = arguments[name];
            if (token == null || token.Type == JTokenType.Null)
            {
                return fallback;
            }

            if (!WireJson.TryGetInteger(token, out var value) || value < min || value > max)
            {
                throw HostError.InvalidArguments(name, $"'{name}' must be an integer from {min} to {max}.");
            }

            return (int)value;
        }

        private static string? ReadSnapshot(JObject arguments)
        {
            var token = arguments["snapshot"];
            if (token == null || token.Type == JTokenType.Null)
            {
                return null;
            }

            var text = token.Type == JTokenType.String ? (string?)token : null;
            if (text == null || !SnapshotPattern.IsMatch(text))
            {
                throw HostError.InvalidArguments("snapshot", "'snapshot' must be a lowercase SHA-256 hex string.");
            }

            return text;
        }
    }

    /// <summary>
    /// Builds paged results and enforces the result byte budget. Page size is measured on the serialized
    /// page object, so a page never exceeds <c>max_result_bytes</c>.
    /// </summary>
    public static class Pager
    {
        /// <summary>
        /// Pages a full item list. Items are serialized with snake_case names. The snapshot is the SHA-256 of
        /// the full serialized item list. Throws stale_snapshot when the request carries a different snapshot,
        /// and result_too_large when the first item alone exceeds the budget.
        /// </summary>
        public static JObject Page<T>(IReadOnlyList<T> items, PageRequest request, string units, int maxResultBytes)
        {
            var itemTexts = items.Select(item => WireJson.Compact(WireJson.ToToken(item))).ToList();
            var snapshot = ComputeSnapshot(itemTexts);

            if (request.Snapshot != null && !string.Equals(request.Snapshot, snapshot, StringComparison.Ordinal))
            {
                throw HostError.Coded(
                    ErrorCodes.StaleSnapshot,
                    "The list changed since the snapshot was taken. Request the page again without snapshot.",
                    new JObject { ["snapshot"] = WireJson.Value(snapshot) });
            }

            var total = itemTexts.Count;
            var start = Math.Min(request.Offset, total);
            var page = new List<string>();

            for (var index = start; index < total && page.Count < request.Limit; index++)
            {
                page.Add(itemTexts[index]);
                if (PageBytes(page, request, start, total, snapshot, units) <= maxResultBytes)
                {
                    continue;
                }

                if (page.Count == 1)
                {
                    throw HostError.Coded(
                        ErrorCodes.ResultTooLarge,
                        "A single item is larger than the result budget.",
                        new JObject { ["budget_bytes"] = WireJson.Value(maxResultBytes) });
                }

                page.RemoveAt(page.Count - 1);
                break;
            }

            var text = BuildPageText(page, request.Offset, NextOffset(start, page.Count, total), total, snapshot, units);
            if (WireJson.ByteCount(text) > maxResultBytes)
            {
                // Reached for empty pages (offset at or beyond the end) whose paging fields alone exceed the budget.
                throw HostError.Coded(
                    ErrorCodes.ResultTooLarge,
                    "The page is larger than the result budget. Ask for a narrower query.",
                    new JObject { ["budget_bytes"] = WireJson.Value(maxResultBytes) });
            }

            return (JObject)JToken.Parse(text);
        }

        /// <summary>Returns the result unchanged, or throws result_too_large when its compact form exceeds the budget.</summary>
        public static JToken RequireFits(JToken result, int maxResultBytes)
        {
            if (WireJson.ByteCount(WireJson.Compact(result)) > maxResultBytes)
            {
                throw HostError.Coded(
                    ErrorCodes.ResultTooLarge,
                    "The result is larger than the result budget. Ask for a narrower query.",
                    new JObject { ["budget_bytes"] = WireJson.Value(maxResultBytes) });
            }

            return result;
        }

        /// <summary>Lowercase hex SHA-256 of the compact JSON array of the serialized items.</summary>
        public static string ComputeSnapshot(IReadOnlyList<string> serializedItems)
        {
            var bytes = Encoding.UTF8.GetBytes("[" + string.Join(",", serializedItems) + "]");
            using (var sha = SHA256.Create())
            {
                var hash = sha.ComputeHash(bytes);
                var builder = new StringBuilder(hash.Length * 2);
                foreach (var b in hash)
                {
                    builder.Append(b.ToString("x2", CultureInfo.InvariantCulture));
                }

                return builder.ToString();
            }
        }

        private static int PageBytes(List<string> page, PageRequest request, int start, int total, string snapshot, string units) =>
            WireJson.ByteCount(BuildPageText(page, request.Offset, NextOffset(start, page.Count, total), total, snapshot, units));

        private static int? NextOffset(int start, int count, int total) =>
            start + count < total ? start + count : (int?)null;

        private static string BuildPageText(
            IReadOnlyList<string> items,
            int offset,
            int? nextOffset,
            int total,
            string snapshot,
            string units) =>
            "{\"items\":[" + string.Join(",", items) + "]"
            + ",\"offset\":" + offset.ToString(CultureInfo.InvariantCulture)
            + ",\"next_offset\":" + (nextOffset.HasValue ? nextOffset.Value.ToString(CultureInfo.InvariantCulture) : "null")
            + ",\"total\":" + total.ToString(CultureInfo.InvariantCulture)
            + ",\"snapshot\":" + JsonConvert.ToString(snapshot)
            + ",\"units\":" + JsonConvert.ToString(units)
            + "}";
    }
}
