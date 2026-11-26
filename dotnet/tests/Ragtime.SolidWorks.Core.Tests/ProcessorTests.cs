using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ProcessorEnvelopeTests
    {
        [Fact]
        public async Task EveryResponseIsOneLineWithoutRawNewlines()
        {
            var h = new Harness();
            var lines = new List<string>
            {
                await h.SendRawAsync("garbage"),
                await h.SendRawAsync(TestData.Request("status")),
                await h.SendRawAsync(TestData.Request("describe")),
                await h.SendRawAsync(TestData.Request("nope")),
            };

            foreach (var line in lines)
            {
                Assert.DoesNotContain("\n", line);
                Assert.DoesNotContain("\r", line);
            }
        }

        [Fact]
        public async Task SuccessEnvelopeHasTheContractFields()
        {
            var h = new Harness();

            var response = await h.CallAsync("status", id: "abc");

            Assert.Equal(1, (int)response["protocol"]!);
            Assert.Equal("abc", (string?)response["id"]);
            Assert.True((bool)response["ok"]!);
            Assert.Equal("status", (string?)response["operation"]);
            Assert.Equal(TestData.HostVersion, (string?)response["host"]!["host_version"]);
            Assert.True(response.ContainsKey("elapsed_ms"));
            Assert.NotNull(response["result"]);
        }

        [Fact]
        public async Task ElapsedMillisecondsComesFromTheClock()
        {
            var h = new Harness();
            h.Clock.Now = 1000;

            var response = await h.CallAsync("describe");

            Assert.Equal(0, (int)response["elapsed_ms"]!);
        }

        [Fact]
        public async Task UnparseableRequestEchoesNullIdAndOperation()
        {
            var h = new Harness();

            var response = await h.SendAsync("{not json");

            Assert.False((bool)response["ok"]!);
            Assert.Equal(JTokenType.Null, response["id"]!.Type);
            Assert.Equal(JTokenType.Null, response["operation"]!.Type);
            Assert.Equal("invalid_request", (string?)response["error"]!["code"]);
            Assert.Equal(JTokenType.Null, response["host"]!["instance_id"]!.Type);
        }

        [Fact]
        public async Task ProtocolMismatchEchoesIdAndOperation()
        {
            var h = new Harness();
            var line = TestData.Request("status", id: "keep-me").Replace("\"protocol\":1", "\"protocol\":9");

            var response = await h.SendAsync(line);

            Assert.Equal("protocol_mismatch", (string?)response["error"]!["code"]);
            Assert.Equal("keep-me", (string?)response["id"]);
            Assert.Equal("status", (string?)response["operation"]);
        }

        [Fact]
        public async Task UnknownOperationIsReported()
        {
            var h = new Harness();

            var response = await h.CallAsync("features.delete");

            Assert.Equal("unknown_operation", (string?)response["error"]!["code"]);
            Assert.Equal("features.delete", (string?)response["operation"]);
        }

        [Fact]
        public void CoreProducedCodesExistInTheContract()
        {
            // Codes produced by Core's own code paths. Codes the instance manager or Readers produce are
            // checked by the passthrough tests below.
            var producedByCore = new[]
            {
                ErrorCodes.InvalidRequest, ErrorCodes.ProtocolMismatch, ErrorCodes.UnknownOperation,
                ErrorCodes.InvalidArguments, ErrorCodes.WriteNotAllowed, ErrorCodes.InstanceLimit,
                ErrorCodes.DocumentNotFound, ErrorCodes.DocumentNotOpen, ErrorCodes.NotSupportedForDocumentType,
                ErrorCodes.StaleSnapshot, ErrorCodes.ResultTooLarge, ErrorCodes.InternalError,
                ErrorCodes.SolidworksNotRunning, ErrorCodes.VersionMismatch,
            };

            foreach (var code in producedByCore)
            {
                Assert.Contains(code, ErrorCodes.All);
            }
        }
    }

    public class ProcessorDispatchTests
    {
        [Fact]
        public async Task StatusReportsIdleTimeWithoutResettingIt()
        {
            var h = new Harness();
            h.Clock.Now = 2500;
            await h.CallAsync("describe");
            await h.CallAsync("pdm.status");
            var activity = (await h.CallAsync("status"))["result"]!["activity"]!;
            Assert.Equal(0, (int)activity["in_flight"]!);
            Assert.Equal(2, (int)activity["idle_seconds"]!);
            Assert.False((bool)activity["pdm_busy"]!);
        }

        [Fact]
        public async Task WorkResetsIdleAndStatusReflectsPdmBusy()
        {
            var h = new Harness();
            h.Clock.Now = 5000;
            h.Pdm.IsBusy = true;
            Assert.True((bool)(await h.CallAsync("documents.list"))["ok"]!);
            h.Clock.Now = 6900;
            var activity = (await h.CallAsync("status"))["result"]!["activity"]!;
            Assert.Equal(1, (int)activity["idle_seconds"]!);
            Assert.True((bool)activity["pdm_busy"]!);
            h.Pdm.IsBusy = false;
            Assert.False((bool)(await h.CallAsync("status"))["result"]!["activity"]!["pdm_busy"]!);
        }

        [Fact]
        public async Task InFlightWorkBlocksConditionalShutdown()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            using var entered = new ManualResetEventSlim();
            using var release = new ManualResetEventSlim();
            h.Manager.SessionFor("i1").BeforeListOpenDocuments = () => { entered.Set(); release.Wait(); };
            var work = Task.Run(() => h.CallAsync("documents.list"));
            try
            {
                Assert.True(entered.Wait(System.TimeSpan.FromSeconds(10)));
                var activity = (await h.CallAsync("status"))["result"]!["activity"]!;
                Assert.Equal(1, (int)activity["in_flight"]!);
                Assert.Equal(0, (int)activity["idle_seconds"]!);
                var response = await h.CallAsync("host.shutdown", Harness.Args(("if_idle_seconds", Harness.I(0))));
                Assert.Equal("host_busy", (string?)response["error"]!["code"]);
                Assert.Equal(new[] { "request_in_flight" }, ((JArray)response["error"]!["details"]!["reasons"]!).Values<string>().ToArray());
                Assert.Equal(0, h.ShutdownCalls);
            }
            finally
            {
                release.Set();
                await work;
            }
            Assert.True((bool)(await h.CallAsync("documents.list"))["ok"]!);
        }

        [Theory]
        [InlineData("recently_used")]
        [InlineData("instance_busy")]
        [InlineData("documents_open")]
        [InlineData("pdm_busy")]
        public async Task ConditionalShutdownRefusesEachBlockingReason(string reason)
        {
            var h = new Harness();
            if (reason == "instance_busy") h.Manager.AddBackground("i1", busy: true);
            if (reason == "pdm_busy") h.Pdm.IsBusy = true;
            if (reason == "documents_open")
            {
                h.Manager.AddBackground("i1");
                Assert.True((bool)(await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Work\part.sldprt")))))["ok"]!);
            }
            var threshold = reason == "recently_used" ? 1 : 0;
            var response = await h.CallAsync("host.shutdown", Harness.Args(("if_idle_seconds", Harness.I(threshold))));
            Assert.Equal("host_busy", (string?)response["error"]!["code"]);
            Assert.Equal(new[] { reason }, ((JArray)response["error"]!["details"]!["reasons"]!).Values<string>().ToArray());
            Assert.Equal(0, h.ShutdownCalls);
            Assert.True((bool)(await h.CallAsync("documents.list"))["ok"]!);
        }

        [Fact]
        public async Task ConditionalShutdownStopsWorkButLeavesNeutralOperationsAvailable()
        {
            var h = new Harness();
            var response = await h.CallAsync("host.shutdown", Harness.Args(("if_idle_seconds", Harness.I(0))));
            Assert.True((bool)response["result"]!["stopping"]!);
            Assert.Equal(1, h.ShutdownCalls);
            Assert.Equal("host_stopping", (string?)(await h.CallAsync("documents.list"))["error"]!["code"]);
            Assert.True((bool)(await h.CallAsync("status"))["ok"]!);
            Assert.True((bool)(await h.CallAsync("describe"))["ok"]!);
            Assert.True((bool)(await h.CallAsync("host.shutdown", Harness.Args(("if_idle_seconds", Harness.I(86400)))))["result"]!["stopping"]!);
        }

        [Fact]
        public async Task ConditionalShutdownRetriesWhenWorkOpensDocumentAfterSnapshot()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            JObject? opened = null;
            var reads = 0;
            h.Pdm.OnIsBusyRead = () =>
            {
                reads++;
                if (reads == 1)
                {
                    opened = h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Work\part.sldprt"))))
                        .GetAwaiter().GetResult();
                }
            };

            var response = await h.CallAsync("host.shutdown", Harness.Args(("if_idle_seconds", Harness.I(0))));

            Assert.NotNull(opened);
            Assert.True((bool)opened!["ok"]!);
            Assert.Equal(2, reads);
            Assert.Equal("host_busy", (string?)response["error"]!["code"]);
            Assert.Equal(new[] { "documents_open" }, ((JArray)response["error"]!["details"]!["reasons"]!).Values<string>().ToArray());
            Assert.Equal(0, h.ShutdownCalls);
        }

        [Fact]
        public async Task ConditionalShutdownRefusesAfterThreeChangingSnapshots()
        {
            var h = new Harness();
            var reads = 0;
            h.Pdm.OnIsBusyRead = () =>
            {
                reads++;
                h.CallAsync("documents.list").GetAwaiter().GetResult();
            };

            var response = await h.CallAsync("host.shutdown", Harness.Args(("if_idle_seconds", Harness.I(0))));

            Assert.Equal(3, reads);
            Assert.Equal("host_busy", (string?)response["error"]!["code"]);
            Assert.Equal(new[] { "request_in_flight" }, ((JArray)response["error"]!["details"]!["reasons"]!).Values<string>().ToArray());
            Assert.Equal(0, h.ShutdownCalls);
        }

        [Fact]
        public async Task PlainShutdownStopsWorkAndCanBeRepeated()
        {
            var h = new Harness();
            Assert.True((bool)(await h.CallAsync("host.shutdown"))["result"]!["stopping"]!);
            Assert.Equal("host_stopping", (string?)(await h.CallAsync("documents.list"))["error"]!["code"]);
            Assert.True((bool)(await h.CallAsync("host.shutdown"))["result"]!["stopping"]!);
        }

        [Theory]
        [InlineData(-1)]
        [InlineData(86401)]
        public async Task ConditionalShutdownRejectsInvalidThreshold(int threshold)
        {
            var h = new Harness();
            var response = await h.CallAsync("host.shutdown", Harness.Args(("if_idle_seconds", Harness.I(threshold))));
            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
            Assert.Equal(0, h.ShutdownCalls);
        }

        [Fact]
        public async Task DescribeOmitsHostShutdownAndListsEverySectionFiveOperation()
        {
            var h = new Harness();

            var response = await h.CallAsync("describe");

            var names = ((JArray)response["result"]!["operations"]!).Select(o => (string?)o["name"]).ToList();
            Assert.DoesNotContain("host.shutdown", names);
            Assert.Equal(33, names.Count);
            Assert.Contains("features.list", names);
            Assert.Contains("mass_properties.get", names);
        }

        [Fact]
        public async Task StatusReportsAccountSessionAndInstances()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1", documents: 0);

            var status = (await h.CallAsync("status"))["result"]!;

            Assert.Equal("DESK\\elfy", (string?)status["account"]!["name"]);
            Assert.Equal(TestData.Sid, (string?)status["account"]!["sid"]);
            Assert.False((bool)status["account"]!["is_administrator"]!);
            Assert.True((bool)status["session"]!["interactive"]!);
            Assert.Equal(1, (int)status["session"]!["id"]!);
            Assert.Equal(42.5, (double)status["uptime_seconds"]!);
            Assert.Equal(2024, (int)status["solidworks"]!["installed_versions"]![0]!);
            var instance = status["solidworks"]!["instances"]![0]!;
            Assert.Equal("i1", (string?)instance["instance_id"]);
            Assert.Equal(100, (int)instance["pid"]!);
            Assert.Equal(0, (int)instance["documents"]!);
        }

        [Fact]
        public async Task HostShutdownCallsCallbackAndAnswersStopping()
        {
            var h = new Harness();

            var response = await h.CallAsync("host.shutdown");

            Assert.True((bool)response["ok"]!);
            Assert.True((bool)response["result"]!["stopping"]!);
            Assert.Equal(1, h.ShutdownCalls);
        }

        [Fact]
        public async Task DocumentsListIsEmptyWithoutInstances()
        {
            var h = new Harness();

            var result = (await h.CallAsync("documents.list"))["result"]!;

            Assert.Empty((JArray)result["items"]!);
            Assert.Equal(0, (int)result["total"]!);
            Assert.Equal(JTokenType.Null, result["next_offset"]!.Type);
            Assert.Equal("SI: m, kg, s, rad", (string?)result["units"]);
        }

        [Fact]
        public async Task DocumentOpenStartsAnInstanceAndReturnsTheItem()
        {
            var h = new Harness();

            var response = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            Assert.True((bool)response["ok"]!);
            var item = response["result"]!;
            Assert.Equal("doc-1", (string?)item["document_id"]);
            Assert.Equal("i1", (string?)item["instance_id"]);
            Assert.True((bool)item["opened_by_host"]!);
            Assert.Equal("part", (string?)item["type"]);
            Assert.Equal("i1", (string?)response["host"]!["instance_id"]);
            Assert.Equal("32.5.0", (string?)response["host"]!["solidworks_revision"]);
            Assert.Equal(2024, (int)response["host"]!["solidworks_version"]!);
            Assert.Equal(new[] { "StartNew" }, h.Manager.Targets.Where(t => t.StartsWith("StartNew", System.StringComparison.Ordinal)).ToArray());
        }

        [Fact]
        public async Task SecondOpenReusesIdleInstanceAndSamePathKeepsId()
        {
            var h = new Harness();
            var args = Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt")));
            await h.CallAsync("document.open", args);

            var again = await h.CallAsync("document.open", args);

            Assert.Equal("doc-1", (string?)again["result"]!["document_id"]);
            Assert.Single(h.Manager.Instances);
        }

        [Fact]
        public async Task SecondOpenStartsAnotherInstanceWhenPoolHasRoom()
        {
            var h = new Harness();
            var context = TestData.Context(maxInstances: 2);
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\A.sldprt"))), context);

            var second = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\B.sldprt"))), context);

            Assert.Equal("i2", (string?)second["result"]!["instance_id"]);
            Assert.Equal(2, h.Manager.Instances.Count);
        }

        [Fact]
        public async Task OpenWithAllInstancesBusyIsInstanceLimit()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1", documents: 1, busy: true);

            var response = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\A.sldprt"))));

            Assert.Equal("instance_limit", (string?)response["error"]!["code"]);
            Assert.Equal(1, (int)response["error"]!["details"]!["max_instances"]!);
        }

        [Fact]
        public async Task OpenRejectsUncAndRelativePaths()
        {
            var h = new Harness();

            var unc = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"\\server\share\part.sldprt"))));
            var relative = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"part.sldprt"))));

            Assert.Equal("invalid_arguments", (string?)unc["error"]!["code"]);
            Assert.Equal("path", (string?)unc["error"]!["details"]!["field"]);
            Assert.Equal("invalid_arguments", (string?)relative["error"]!["code"]);
            Assert.Empty(h.Manager.Targets);
        }

        [Fact]
        public async Task OpenIsRefusedInAttachMode()
        {
            var h = new Harness();
            h.Manager.AddAttached(100);

            var response = await h.CallAsync(
                "document.open",
                Harness.Args(("path", Harness.S(@"C:\Models\A.sldprt"))),
                TestData.Context(mode: "attach"));

            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task MissingFileIsDocumentNotFoundFromTheSession()
        {
            var h = new Harness();

            var response = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\missing.sldprt"))));

            Assert.Equal("document_not_found", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task ModelOperationWithIdResolvesAndPassesThroughPaging()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));
            h.Manager.SessionFor("i1").FeatureCount = 5;

            var first = await h.CallAsync("features.list", Harness.Args(("document", Harness.S("doc-1")), ("limit", Harness.I(2))));
            var second = await h.CallAsync("features.list", Harness.Args(("document", Harness.S("doc-1")), ("offset", Harness.I(2)), ("limit", Harness.I(2))));

            Assert.Equal(2, (int)first["result"]!["next_offset"]!);
            Assert.Equal(5, (int)first["result"]!["total"]!);
            Assert.Equal(4, (int)second["result"]!["next_offset"]!);
            Assert.Equal("Boss-Extrude1", (string?)first["result"]!["items"]![0]!["name"]);
        }

        [Fact]
        public async Task ModelOperationAcceptsFullPathOfOpenDocument()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var response = await h.CallAsync("model.summary", Harness.Args(("document", Harness.S(@"c:\models\BRACKET.sldprt"))));

            Assert.True((bool)response["ok"]!);
            Assert.Equal("doc-1", (string?)response["result"]!["document_id"]);
            Assert.Equal(1, (int)response["result"]!["solid_body_count"]!);
        }

        [Fact]
        public async Task PathOfUnopenedDocumentIsDocumentNotOpen()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");

            var response = await h.CallAsync("model.summary", Harness.Args(("document", Harness.S(@"C:\Models\Nope.sldprt"))));

            Assert.Equal("document_not_open", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task UnknownDocumentIdIsDocumentNotFound()
        {
            var h = new Harness();

            var response = await h.CallAsync("model.summary", Harness.Args(("document", Harness.S("doc-7"))));

            Assert.Equal("document_not_found", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task DocumentAlreadyOpenInSolidworksIsFoundByRefreshAndCannotBeClosed()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            h.Manager.SessionFor("i1").Open.Add(new SessionDocumentInfo("User.sldprt", @"C:\Models\User.sldprt", "part", "Default", false, true));

            var listed = (await h.CallAsync("documents.list"))["result"]!;
            var close = await h.CallAsync("document.close", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.Equal(1, (int)listed["total"]!);
            Assert.False((bool)listed["items"]![0]!["opened_by_host"]!);
            Assert.Equal("invalid_arguments", (string?)close["error"]!["code"]);
            Assert.DoesNotContain("CloseDocument", h.Manager.SessionFor("i1").Calls);
        }

        [Fact]
        public async Task DocumentCloseRemovesTheDocument()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var close = await h.CallAsync("document.close", Harness.Args(("document", Harness.S("doc-1"))));
            var after = await h.CallAsync("model.summary", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.True((bool)close["result"]!["closed"]!);
            Assert.Equal(new[] { @"C:\Models\Bracket.sldprt" }, h.Manager.SessionFor("i1").Closed);
            Assert.Equal("document_not_found", (string?)after["error"]!["code"]);
        }

        [Fact]
        public async Task MatesOnPartIsNotSupportedForDocumentType()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var response = await h.CallAsync("mates.list", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.Equal("not_supported_for_document_type", (string?)response["error"]!["code"]);
            Assert.Equal("part", (string?)response["error"]!["details"]!["document_type"]);
            Assert.DoesNotContain("ListMates", h.Manager.SessionFor("i1").Calls);
        }

        [Fact]
        public async Task ComponentsOnPartIsNotSupportedForDocumentType()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var response = await h.CallAsync("components.list", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.Equal("not_supported_for_document_type", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task MatesOnAssemblyWork()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Asm.sldasm"))));

            var mates = await h.CallAsync("mates.list", Harness.Args(("document", Harness.S("doc-1"))));
            var components = await h.CallAsync("components.list", Harness.Args(("document", Harness.S("doc-1")), ("top_level_only", WireJson.Value(false))));

            Assert.True((bool)mates["ok"]!);
            Assert.Equal("Coincident1", (string?)mates["result"]!["items"]![0]!["name"]);
            Assert.True((bool)components["ok"]!);
            Assert.Contains("ListComponents:False", h.Manager.SessionFor("i1").Calls);
        }

        [Fact]
        public async Task SketchSummaryAndPartsAndUnknownSketch()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var summary = await h.CallAsync("sketch.get", Harness.Args(("document", Harness.S("doc-1")), ("sketch", Harness.S("Sketch1"))));
            var segments = await h.CallAsync("sketch.get", Harness.Args(("document", Harness.S("doc-1")), ("sketch", Harness.S("Sketch1")), ("part", Harness.S("segments")), ("limit", Harness.I(2))));
            var missing = await h.CallAsync("sketch.get", Harness.Args(("document", Harness.S("doc-1")), ("sketch", Harness.S("Nope"))));

            Assert.Equal("Sketch1", (string?)summary["result"]!["name"]);
            Assert.Equal(2, ((JArray)segments["result"]!["items"]!).Count);
            Assert.Equal(3, (int)segments["result"]!["total"]!);
            Assert.Equal("entity_not_found", (string?)missing["error"]!["code"]);
        }

        [Fact]
        public async Task FeatureGetPassesThroughEntityNotFound()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var response = await h.CallAsync("feature.get", Harness.Args(("document", Harness.S("doc-1")), ("feature", Harness.S("Nope"))));

            Assert.Equal("entity_not_found", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task StaleSnapshotReturnsCurrentHash()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));
            var page = await h.CallAsync("features.list", Harness.Args(("document", Harness.S("doc-1"))));
            var current = (string?)page["result"]!["snapshot"];
            h.Manager.SessionFor("i1").FeatureCount = 4;

            var response = await h.CallAsync("features.list", Harness.Args(("document", Harness.S("doc-1")), ("snapshot", Harness.S(current!))));

            Assert.Equal("stale_snapshot", (string?)response["error"]!["code"]);
            Assert.NotEqual(current, (string?)response["error"]!["details"]!["snapshot"]);
        }

        [Fact]
        public async Task MatchingSnapshotPagesNormally()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));
            var page = await h.CallAsync("features.list", Harness.Args(("document", Harness.S("doc-1"))));
            var snapshot = (string?)page["result"]!["snapshot"];

            var next = await h.CallAsync("features.list", Harness.Args(("document", Harness.S("doc-1")), ("snapshot", Harness.S(snapshot!)), ("offset", Harness.I(1))));

            Assert.True((bool)next["ok"]!);
        }

        [Fact]
        public async Task NonPagedResultOverBudgetIsResultTooLarge()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var response = await h.CallAsync(
                "mass_properties.get",
                Harness.Args(("document", Harness.S("doc-1"))),
                TestData.Context(maxBytes: 40));

            Assert.Equal("result_too_large", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task PagedItemOverBudgetIsResultTooLarge()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var response = await h.CallAsync("features.list", Harness.Args(("document", Harness.S("doc-1"))), TestData.Context(maxBytes: 60));

            Assert.Equal("result_too_large", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task WriteOperationIsRefusedWithoutAllowWriteAndSessionIsNotCalled()
        {
            var registry = new OperationRegistry(new[]
            {
                new OperationDefinition("test.write", "Synthetic write.", ProtocolInfo.Access.Write, false, new ObjectSchema().Build(), "r"),
            });
            var h = new Harness(registry);

            var response = await h.CallAsync("test.write", context: TestData.Context(allowWrite: false));

            Assert.Equal("write_not_allowed", (string?)response["error"]!["code"]);
            Assert.Empty(h.Manager.Targets);
        }

        [Fact]
        public async Task WriteOperationPassesTheGateWhenAllowed()
        {
            var registry = new OperationRegistry(new[]
            {
                new OperationDefinition("test.write", "Synthetic write.", ProtocolInfo.Access.Write, false, new ObjectSchema().Build(), "r"),
            });
            var h = new Harness(registry);

            var response = await h.CallAsync("test.write", context: TestData.Context(allowWrite: true));

            Assert.NotEqual("write_not_allowed", (string?)response["error"]?["code"]);
        }

        [Fact]
        public async Task DimensionsForOneFeatureSucceeds()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var response = await h.CallAsync("dimensions.list", Harness.Args(("document", Harness.S("doc-1")), ("feature", Harness.S("Boss-Extrude1"))));

            Assert.True((bool)response["ok"]!);
            Assert.Equal("Boss-Extrude1", (string?)response["result"]!["items"]![0]!["feature"]);
        }

        [Fact]
        public async Task UnexpectedExceptionBecomesInternalErrorWithoutStackOrMessage()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));
            h.Manager.SessionFor("i1").ThrowUnexpected = true;

            var raw = await h.SendRawAsync(TestData.Request("features.list", Harness.Args(("document", Harness.S("doc-1")))));
            var response = JObject.Parse(raw);

            Assert.Equal("internal_error", (string?)response["error"]!["code"]);
            Assert.Equal("InvalidOperationException", (string?)response["error"]!["details"]!["exception_type"]);
            Assert.DoesNotContain("secret detail", raw);
            Assert.DoesNotContain("C:\\private", raw);
            Assert.DoesNotContain("   at ", raw);
            Assert.Single(h.Unexpected);
        }

        [Fact]
        public async Task InstanceStoppedForgetsItsDocuments()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            h.Manager.Stop("i1");
            var listed = (await h.CallAsync("documents.list"))["result"]!;

            Assert.Equal(0, (int)listed["total"]!);
        }

        [Fact]
        public async Task StatusCountsDocumentsFromTheRegistry()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var status = (await h.CallAsync("status"))["result"]!;

            Assert.Equal(1, (int)status["solidworks"]!["instances"]![0]!["documents"]!);
        }

        [Fact]
        public async Task AttachModeListsTheAttachedInstance()
        {
            var h = new Harness();
            h.Manager.AddAttached(500);
            h.Manager.SessionFor("a500").Open.Add(new SessionDocumentInfo("Live.sldasm", @"C:\Models\Live.sldasm", "assembly", null, false, false));
            var context = TestData.Context(mode: "attach");

            var listed = await h.CallAsync("documents.list", context: context);
            var mates = await h.CallAsync("mates.list", Harness.Args(("document", Harness.S(@"C:\Models\Live.sldasm"))), context);

            Assert.Equal(1, (int)listed["result"]!["total"]!);
            Assert.Equal("assembly", (string?)listed["result"]!["items"]![0]!["type"]);
            Assert.True((bool)mates["ok"]!);
            Assert.Equal(1, h.Manager.Targets.Count(t => t == "Attach"));
            Assert.Equal("Specific:a500", h.Manager.Targets[h.Manager.Targets.Count - 1]);
        }

        [Fact]
        public async Task AttachModeWithoutSolidworksIsSolidworksNotRunning()
        {
            var h = new Harness();

            var response = await h.CallAsync("documents.list", context: TestData.Context(mode: "attach"));

            Assert.Equal("solidworks_not_running", (string?)response["error"]!["code"]);
        }

        [Fact]
        public async Task ProcessorNeverThrowsOnNullOrEmptyLines()
        {
            var h = new Harness();

            var empty = await h.SendAsync(string.Empty);
            var spaces = await h.SendAsync("   ");

            Assert.Equal("invalid_request", (string?)empty["error"]!["code"]);
            Assert.Equal("invalid_request", (string?)spaces["error"]!["code"]);
        }

        [Fact]
        public async Task HostShutdownWithCancelledTokenStillProducesEnvelope()
        {
            var h = new Harness();
            var cts = new CancellationTokenSource();
            cts.Cancel();

            var line = await h.Processor.ProcessLineAsync(
                TestData.Request("host.shutdown"),
                cts.Token);

            Assert.NotEmpty(line);
            Assert.Contains("\"ok\":", line);
        }
    }
}
