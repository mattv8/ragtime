using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>
    /// The diagnostics of contract §5.5 through the processor: mates.list (only_problems, error objects, entities),
    /// components.list (file_exists, suppression, only_problems), model.summary (missing_component_count) and document.open
    /// (load_warnings, missing_references).
    /// </summary>
    public class DiagnosticsTests
    {
        private const string AssemblyPath = @"C:\Models\Asm.sldasm";

        /// <summary>A harness with one background instance holding the assembly, opened as doc-1.</summary>
        private static async Task<Harness> AssemblyHarness(System.Action<FakeSession>? seed = null)
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            seed?.Invoke(h.Manager.SessionFor("i1"));
            var opened = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(AssemblyPath))));
            Assert.True((bool)opened["ok"]!, opened.ToString());
            return h;
        }

        private static JToken Result(JObject response) => response["result"]!;

        private static MateItem HealthyMate(string name) => new MateItem { Name = name, Type = "Coincident", Alignment = "Aligned" };

        private static MateItem BrokenMate() => new MateItem
        {
            Name = "Mate-Broken",
            Type = "Coincident",
            Alignment = "Aligned",
            ErrorCode = "47",
            WhatsWrong = "The mate cannot be solved as defined.",
            Entities = new System.Collections.Generic.List<MateEntity>
            {
                new MateEntity { Component = "Sub-1", ComponentPath = "Sub-1", EntityType = "face", Resolved = true },
                new MateEntity { Component = "", ComponentPath = null, EntityType = "unknown", Resolved = false },
            },
        };

        [Fact]
        public async Task MateListReportsTheErrorObjectWhatsWrongAndEntityFields()
        {
            var h = await AssemblyHarness(session =>
            {
                session.MateList.Clear();
                session.MateList.Add(BrokenMate());
            });

            var item = (JObject)Result(await h.CallAsync("mates.list", Harness.Args(("document", Harness.S("doc-1")))))["items"]![0]!;

            Assert.Equal("47", (string?)item["error"]!["code"]);
            Assert.Equal("swFeatureErrorMateIlldefined", (string?)item["error"]!["name"]);
            Assert.Equal("The mate cannot be solved as defined.", (string?)item["whats_wrong"]);
            var entities = (JArray)item["entities"]!;
            Assert.Equal("Sub-1", (string?)entities[0]["component_path"]);
            Assert.True((bool)entities[0]["resolved"]!);
            Assert.Equal(JTokenType.Null, entities[1]["component_path"]!.Type);
            Assert.False((bool)entities[1]["resolved"]!);
            Assert.Equal("unknown", (string?)entities[1]["entity_type"]);
        }

        [Fact]
        public async Task HealthyMateHasNullErrorAndWhatsWrongOnTheWire()
        {
            var h = await AssemblyHarness(session =>
            {
                session.MateList.Clear();
                session.MateList.Add(HealthyMate("Coincident1"));
            });

            var line = await h.SendRawAsync(TestData.Request("mates.list", Harness.Args(("document", Harness.S("doc-1")))));

            Assert.Contains("\"error\":null", line);
            Assert.Contains("\"whats_wrong\":null", line);
        }

        [Fact]
        public async Task OnlyProblemsKeepsErrorsWarningsAndUnresolvedEntitiesOnly()
        {
            var h = await AssemblyHarness(session =>
            {
                session.MateList.Clear();
                session.MateList.Add(HealthyMate("Healthy"));
                session.MateList.Add(BrokenMate());
                session.MateList.Add(new MateItem { Name = "Warned", Type = "Distance", Alignment = "Aligned", Warning = "51" });
                session.MateList.Add(new MateItem
                {
                    Name = "Dangling",
                    Type = "Coincident",
                    Alignment = "Aligned",
                    Entities = new System.Collections.Generic.List<MateEntity> { new MateEntity { Component = "Sub-2", EntityType = "edge", Resolved = false } },
                });
            });

            var all = Result(await h.CallAsync("mates.list", Harness.Args(("document", Harness.S("doc-1")))));
            var problems = Result(await h.CallAsync("mates.list", Harness.Args(("document", Harness.S("doc-1")), ("only_problems", Harness.B(true)))));

            Assert.Equal(4, ((JArray)all["items"]!).Count);
            Assert.Equal(new[] { "Mate-Broken", "Warned", "Dangling" }, ((JArray)problems["items"]!).Select(item => (string?)item["name"]));
        }

        [Fact]
        public async Task ComponentsReportFileExistsAndSuppressionAndOnlyProblemsKeepsMissingFiles()
        {
            var h = await AssemblyHarness(session =>
            {
                session.ComponentList.Clear();
                session.ComponentList.Add(new ComponentItem { Name = "Present-1", FileExists = true, Suppression = ComponentSuppression.Lightweight, Transform = new double[16] });
                session.ComponentList.Add(new ComponentItem { Name = "Gone-1", FileExists = false, Suppression = ComponentSuppression.Resolved, Transform = new double[16] });
            });

            var all = (JArray)Result(await h.CallAsync("components.list", Harness.Args(("document", Harness.S("doc-1")))))["items"]!;
            var problems = (JArray)Result(await h.CallAsync("components.list", Harness.Args(("document", Harness.S("doc-1")), ("only_problems", Harness.B(true)))))["items"]!;

            Assert.True((bool)all[0]["file_exists"]!);
            Assert.Equal("lightweight", (string?)all[0]["suppression"]);
            Assert.False((bool)all[1]["file_exists"]!);
            Assert.Equal("missing_file", (string?)all[1]["suppression"]);
            Assert.Single(problems);
            Assert.Equal("Gone-1", (string?)problems[0]["name"]);
        }

        [Fact]
        public async Task AssemblySummaryCountsComponentsWhoseFileIsMissing()
        {
            var h = await AssemblyHarness(session =>
            {
                session.ComponentList.Clear();
                session.ComponentList.Add(new ComponentItem { Name = "Present-1", FileExists = true, Transform = new double[16] });
                session.ComponentList.Add(new ComponentItem { Name = "Gone-1", FileExists = false, Transform = new double[16] });
                session.ComponentList.Add(new ComponentItem { Name = "Gone-2", FileExists = false, Transform = new double[16] });
            });

            var summary = Result(await h.CallAsync("model.summary", Harness.Args(("document", Harness.S("doc-1")))));

            Assert.Equal(2, (int)summary["missing_component_count"]!);
        }

        [Fact]
        public async Task PartSummaryHasNoMissingComponentCount()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var summary = Result(await h.CallAsync("model.summary", Harness.Args(("document", Harness.S("doc-1")))));

            Assert.False(summary.ToObject<JObject>()!.ContainsKey("missing_component_count"));
        }

        [Fact]
        public async Task DocumentOpenNamesTheLoadWarningsAndListsMissingReferences()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            var session = h.Manager.SessionFor("i1");
            session.LoadWarningBits = 0x100000 | 0x2;
            session.MissingReferences.Add(@"C:\Parts\Gone.sldprt");

            var opened = Result(await h.CallAsync("document.open", Harness.Args(("path", Harness.S(AssemblyPath)))));

            Assert.Equal(new[] { "read_only", "references_not_found" }.OrderBy(name => name).ToArray(),
                ((JArray)opened["load_warnings"]!).Select(name => (string?)name).OrderBy(name => name).ToArray());
            Assert.Equal(new[] { @"C:\Parts\Gone.sldprt" }, ((JArray)opened["missing_references"]!).Select(path => (string?)path));
        }

        [Fact]
        public async Task DocumentOpenCapsMissingReferencesAtFifty()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            var session = h.Manager.SessionFor("i1");
            for (var index = 0; index < 60; index++)
            {
                session.MissingReferences.Add(@"C:\Parts\P" + index + ".sldprt");
            }

            var opened = Result(await h.CallAsync("document.open", Harness.Args(("path", Harness.S(AssemblyPath)))));

            Assert.Equal(50, ((JArray)opened["missing_references"]!).Count);
            Assert.Empty((JArray)opened["load_warnings"]!);
        }

        [Fact]
        public async Task DocumentListStillOmitsTheOpenOnlyFields()
        {
            var h = await AssemblyHarness();

            var listed = Result(await h.CallAsync("documents.list"));

            Assert.False(((JObject)listed["items"]![0]!).ContainsKey("load_warnings"));
        }
    }
}
