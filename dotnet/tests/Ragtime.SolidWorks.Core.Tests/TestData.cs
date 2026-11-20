using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>Builders for request lines and fake clocks used across the tests.</summary>
    internal static class TestData
    {
        public const string Sid = "S-1-5-21-1111111111-2222222222-3333333333-1001";
        public const string HostVersion = "0.1.0";
        public const string ToolId = "test-tool";
        public const string WorkFolder = @"C:\Work";

        public static JObject Context(
            int? version = 2024,
            string mode = "background",
            int maxInstances = 1,
            int idle = 900,
            bool allowWrite = false,
            int deadline = 90,
            int maxBytes = 32768)
        {
            return new JObject
            {
                ["solidworks_version"] = WireJson.Value(version),
                ["instance_mode"] = WireJson.Value(mode),
                ["max_instances"] = WireJson.Value(maxInstances),
                ["idle_shutdown_seconds"] = WireJson.Value(idle),
                ["allow_write"] = WireJson.Value(allowWrite),
                ["deadline_seconds"] = WireJson.Value(deadline),
                ["max_result_bytes"] = WireJson.Value(maxBytes),
            };
        }

        /// <summary>Comment prefix Ragtime sends in <c>context.pdm</c> for these tests.</summary>
        public const string CommentPrefix = "[Ragtime: Test; user: tester] ";

        /// <summary>A request context with <c>allow_write</c> and a linked PDM vault (<c>context.pdm</c>).</summary>
        public static JObject LinkedContext(bool allowWrite = true, int? version = 2024, IEnumerable<string>? approved = null)
        {
            var context = Context(version: version, allowWrite: allowWrite);
            context["pdm"] = new JObject
            {
                ["tool_id"] = ToolId,
                ["vault"] = WireJson.Value(FakePdmVault.VaultName),
                ["approved_folders"] = JArray.FromObject((approved ?? new[] { @"\13_Ragtime" }).ToList()),
                ["comment_prefix"] = WireJson.Value(CommentPrefix),
            };
            return context;
        }

        public static string Request(string operation, JObject? arguments = null, JObject? context = null, string id = "req-1")
        {
            var root = new JObject
            {
                ["protocol"] = WireJson.Value(1),
                ["id"] = WireJson.Value(id),
                ["operation"] = WireJson.Value(operation),
                ["arguments"] = arguments ?? new JObject(),
                ["context"] = context ?? Context(),
            };
            return root.ToString(Newtonsoft.Json.Formatting.None);
        }

        public static RequestContext DefaultContext() =>
            new RequestContext(2024, "background", 1, 900, false, 90, 32768);
    }

    /// <summary>A clock the test moves by hand.</summary>
    internal sealed class FakeClock : IMonotonicClock
    {
        public long Now { get; set; }

        public long ElapsedMilliseconds => Now;

        public void Advance(TimeSpan span) => Now += (long)span.TotalMilliseconds;
    }
}
