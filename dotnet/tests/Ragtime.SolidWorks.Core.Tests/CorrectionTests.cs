using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>C1: one deadline per request, shared by every instance call of that request.</summary>
    public class ProcessorDeadlineTests
    {
        [Fact]
        public async Task DocumentsListGivesEveryInstanceCallTheSameDeadline()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1", documents: 1);
            h.Manager.AddBackground("i2", documents: 1);
            h.Manager.CostMs = 3000;

            var listed = await h.CallAsync("documents.list", context: TestData.Context(maxInstances: 2, deadline: 5));

            Assert.True((bool)listed["ok"]!);
            Assert.Equal(2, h.Manager.Deadlines.Count);
            Assert.Same(h.Manager.Deadlines[0], h.Manager.Deadlines[1]);
        }

        [Fact]
        public async Task DocumentsListStopsSchedulingWhenTheBudgetIsSpent()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1", documents: 1);
            h.Manager.AddBackground("i2", documents: 1);
            h.Manager.CostMs = 6000;

            var listed = await h.CallAsync("documents.list", context: TestData.Context(maxInstances: 2, deadline: 5));

            Assert.Equal("operation_timeout", (string?)listed["error"]!["code"]);
            Assert.Single(h.Manager.Targets);
        }

        [Fact]
        public async Task RefreshAndOperationShareOneBudget()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            h.Manager.SessionFor("i1").Open.Add(new SessionDocumentInfo("Bracket.sldprt", @"C:\Models\Bracket.sldprt", "part", "Default", true, false));
            h.Manager.CostMs = 3000;

            var summary = await h.CallAsync(
                "model.summary",
                Harness.Args(("document", Harness.S("doc-1"))),
                TestData.Context(deadline: 5));

            Assert.True((bool)summary["ok"]!);
            Assert.Equal(2, h.Manager.Deadlines.Count);
            Assert.Same(h.Manager.Deadlines[0], h.Manager.Deadlines[1]);
        }

        [Fact]
        public async Task RefreshThatSpendsTheBudgetLeavesTheOperationUnstarted()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            h.Manager.SessionFor("i1").Open.Add(new SessionDocumentInfo("Bracket.sldprt", @"C:\Models\Bracket.sldprt", "part", "Default", true, false));
            h.Manager.CostMs = 6000;

            var summary = await h.CallAsync(
                "model.summary",
                Harness.Args(("document", Harness.S("doc-1"))),
                TestData.Context(deadline: 5));

            Assert.Equal("operation_timeout", (string?)summary["error"]!["code"]);
            Assert.Single(h.Manager.Targets);
            Assert.DoesNotContain("GetModelSummary", h.Manager.SessionFor("i1").Calls);
        }
    }

    /// <summary>C2: a request sees and addresses only the instances of its own mode.</summary>
    public class ProcessorModeScopeTests
    {
        private static JObject AttachContext() => TestData.Context(mode: "attach");

        [Fact]
        public async Task AttachRequestsCannotCloseOrReadABackgroundDocument()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));
            h.Manager.AddAttached(500);

            var close = await h.CallAsync("document.close", Harness.Args(("document", Harness.S("doc-1"))), AttachContext());
            var summary = await h.CallAsync("model.summary", Harness.Args(("document", Harness.S("doc-1"))), AttachContext());
            var listed = (await h.CallAsync("documents.list"))["result"]!;

            Assert.Equal("invalid_arguments", (string?)close["error"]!["code"]);
            Assert.Equal("document_not_found", (string?)summary["error"]!["code"]);
            Assert.DoesNotContain("CloseDocument", h.Manager.SessionFor("i1").Calls);
            Assert.DoesNotContain("GetModelSummary", h.Manager.SessionFor("i1").Calls);
            Assert.Equal(1, (int)listed["total"]!);
            Assert.True((bool)listed["items"]![0]!["opened_by_host"]!);
        }

        [Fact]
        public async Task CloseInAttachModeIsRefusedBeforeAnyLookup()
        {
            var h = new Harness();
            h.Manager.AddAttached(500);

            var close = await h.CallAsync("document.close", Harness.Args(("document", Harness.S("doc-9"))), AttachContext());

            Assert.Equal("invalid_arguments", (string?)close["error"]!["code"]);
            Assert.Equal("document", (string?)close["error"]!["details"]!["field"]);
            Assert.Empty(h.Manager.Targets);
        }

        [Fact]
        public async Task AttachRefreshKeepsBackgroundDocumentsAndTheirOpenedByHostFlag()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));
            h.Manager.AddAttached(500);

            var attachList = (await h.CallAsync("documents.list", context: AttachContext()))["result"]!;
            var background = (await h.CallAsync("documents.list"))["result"]!;

            Assert.Equal(0, (int)attachList["total"]!);
            Assert.Equal(1, (int)background["total"]!);
            Assert.Equal("doc-1", (string?)background["items"]![0]!["document_id"]);
            Assert.True((bool)background["items"]![0]!["opened_by_host"]!);
        }

        [Fact]
        public async Task SwitchingBackToBackgroundStillClosesTheDocument()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));
            h.Manager.AddAttached(500);
            await h.CallAsync("documents.list", context: AttachContext());

            var close = await h.CallAsync("document.close", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.True((bool)close["ok"]!);
            Assert.Equal(new[] { @"C:\Models\Bracket.sldprt" }, h.Manager.SessionFor("i1").Closed);
        }

        [Fact]
        public async Task BackgroundListDoesNotDropAttachedDocuments()
        {
            var h = new Harness();
            h.Manager.AddAttached(500);
            h.Manager.SessionFor("a500").Open.Add(new SessionDocumentInfo("Live.sldasm", @"C:\Models\Live.sldasm", "assembly", null, false, false));
            await h.CallAsync("documents.list", context: AttachContext());

            var background = (await h.CallAsync("documents.list"))["result"]!;
            var attachAgain = (await h.CallAsync("documents.list", context: AttachContext()))["result"]!;

            Assert.Equal(0, (int)background["total"]!);
            Assert.Equal(1, (int)attachAgain["total"]!);
        }
    }

    /// <summary>C3: an unsaved document has no path; it is identified by its id and title.</summary>
    public class UnsavedDocumentTests
    {
        private static SessionDocumentInfo Unsaved(string title) =>
            new SessionDocumentInfo(title, string.Empty, "part", null, false, true);

        [Fact]
        public void ReferenceOfAnUnsavedDocumentHasANullPath()
        {
            var registry = new DocumentRegistry();
            registry.Register("i1", Unsaved("Part1"), false);
            var second = registry.Register("i1", Unsaved("Part2"), false);

            Assert.Null(second.ToRef().Path);
            Assert.Equal("Part2", second.ToRef().Title);
        }

        [Fact]
        public void TwoUnsavedDocumentsWithDistinctTitlesResolveToThemselvesAcrossRefreshes()
        {
            var registry = new DocumentRegistry();
            registry.Synchronize("i1", new[] { Unsaved("Part1"), Unsaved("Part2") });
            var firstIds = registry.Snapshot().Select(e => e.DocumentId).ToList();

            registry.Synchronize("i1", new[] { Unsaved("Part1"), Unsaved("Part2") });

            Assert.Equal(firstIds, registry.Snapshot().Select(e => e.DocumentId).ToList());
            Assert.Equal("Part1", registry.Find("doc-1")!.Title);
            Assert.Equal("Part2", registry.Find("doc-2")!.Title);
        }

        [Fact]
        public async Task ModelOperationOnAnUnsavedDocumentReachesTheSessionByTitle()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");
            var session = h.Manager.SessionFor("i1");
            session.Open.Add(Unsaved("Part1"));
            session.Open.Add(Unsaved("Part2"));

            var summary = await h.CallAsync("model.summary", Harness.Args(("document", Harness.S("doc-2"))));

            Assert.Equal("doc-2", (string?)summary["result"]!["document_id"]);
            Assert.Equal("Part2", (string?)summary["result"]!["title"]);
            Assert.Null(session.Refs.Single().Path);
            Assert.Equal("Part2", session.Refs.Single().Title);
        }
    }

    /// <summary>C4: the result budget applies to the completed page, including empty pages.</summary>
    public class PagerBudgetTests
    {
        private static List<JObject> NoItems() => new List<JObject>();

        [Fact]
        public void EmptyPageOverTheBudgetIsResultTooLarge()
        {
            var error = Assert.Throws<HostErrorException>(() =>
                Pager.Page(NoItems(), new PageRequest(0, 100, null), "u", 40));

            Assert.Equal(ErrorCodes.ResultTooLarge, error.Code);
        }

        [Fact]
        public void OffsetBeyondTheEndIsCheckedAgainstTheBudget()
        {
            var items = new List<JObject> { new JObject(), new JObject(), new JObject() };

            var error = Assert.Throws<HostErrorException>(() =>
                Pager.Page(items, new PageRequest(50, 100, null), "u", 60));

            Assert.Equal(ErrorCodes.ResultTooLarge, error.Code);
        }

        [Fact]
        public void EmptyPageWithinTheBudgetIsReturned()
        {
            var page = Pager.Page(NoItems(), new PageRequest(0, 100, null), "u", 32768);

            Assert.Empty((JArray)page["items"]!);
        }
    }

    /// <summary>C5: the pool respects the version pin (see InstancePool for the exact rule).</summary>
    public class InstancePoolPinTests
    {
        private static InstanceSnapshot Background(string id, int year, int documents = 0, bool busy = false) =>
            new InstanceSnapshot(id, ProtocolInfo.InstanceModes.Background, true, year, null, 100, false, busy, documents);

        [Fact]
        public void IdleInstanceOfAnotherYearDoesNotBlockStartingTheRequestedYear()
        {
            var choice = InstancePool.ChooseForOpen(new[] { Background("i1", 2024) }, 1, 2023);

            Assert.Equal(PoolAction.StartNew, choice.Action);
        }

        [Fact]
        public void PinnedYearUsesItsIdleInstanceRatherThanAnotherYears()
        {
            var instances = new[] { Background("i1", 2024), Background("i2", 2023) };

            var choice = InstancePool.ChooseForOpen(instances, 2, 2023);

            Assert.Equal(PoolAction.UseExisting, choice.Action);
            Assert.Equal("i2", choice.InstanceId);
        }

        [Fact]
        public void BusyInstanceOfAnotherYearCountsTowardTheLimit()
        {
            var choice = InstancePool.ChooseForOpen(new[] { Background("i1", 2024, busy: true) }, 1, 2023);

            Assert.Equal(PoolAction.LimitReached, choice.Action);
        }

        [Fact]
        public void OtherYearInstanceHoldingDocumentsCountsTowardTheLimit()
        {
            var instances = new[] { Background("i1", 2024, documents: 1) };

            Assert.Equal(PoolAction.LimitReached, InstancePool.ChooseForOpen(instances, 1, 2023).Action);
            Assert.Equal(PoolAction.StartNew, InstancePool.ChooseForOpen(instances, 2, 2023).Action);
        }

        [Fact]
        public void IdleInstancesOfAnotherYearDoNotCountTowardTheLimit()
        {
            var instances = new[] { Background("i1", 2024), Background("i2", 2024) };

            var choice = InstancePool.ChooseForOpen(instances, 1, 2023);

            Assert.Equal(PoolAction.StartNew, choice.Action);
        }

        [Fact]
        public void MatchingInstanceWithDocumentsStartsAnotherWhileRoomRemains()
        {
            var instances = new[] { Background("i1", 2023, documents: 1), Background("i2", 2024) };

            var choice = InstancePool.ChooseForOpen(instances, 2, 2023);

            Assert.Equal(PoolAction.StartNew, choice.Action);
        }

        [Fact]
        public void WithoutAPinEveryBackgroundInstanceIsACandidate()
        {
            var instances = new[] { Background("i1", 2024), Background("i2", 2023) };

            var choice = InstancePool.ChooseForOpen(instances, 1, null);

            Assert.Equal(PoolAction.UseExisting, choice.Action);
            Assert.Equal("i1", choice.InstanceId);
        }

        [Fact]
        public async Task DocumentOpenWithAPinDoesNotUseAnotherYearsIdleInstance()
        {
            var h = new Harness();
            h.Manager.Installed.Add(2023);
            h.Manager.AddBackground("i1");

            var response = await h.CallAsync(
                "document.open",
                Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))),
                TestData.Context(version: 2023));

            Assert.Equal(new[] { "StartNew" }, h.Manager.Targets);
            Assert.True((bool)response["ok"]!);
        }
    }

    /// <summary>F5: the scope of an attach-by-path lookup is rebuilt after the refresh that registers the instance.</summary>
    public class FirstAttachByPathTests
    {
        [Fact]
        public async Task AttachByPathOnAFreshHostSeesTheInstanceItsRefreshRegistered()
        {
            var h = new Harness();
            h.Manager.AddPendingAttached(500, new SessionDocumentInfo("Live.sldasm", @"C:\Models\Live.sldasm", "assembly", null, false, false));

            var response = await h.CallAsync(
                "mates.list",
                Harness.Args(("document", Harness.S(@"C:\Models\Live.sldasm"))),
                TestData.Context(mode: "attach"));

            Assert.True((bool)response["ok"]!);
            Assert.Equal(new[] { "Attach", "Specific:a500" }, h.Manager.Targets);
        }
    }

    /// <summary>F6: the pure rule for the instance that may be closed to make room for a launch.</summary>
    public class InstanceRoomTests
    {
        private static InstanceSnapshot Background(string id, int? year = 2024, int documents = 0, bool busy = false, bool owned = true) =>
            new InstanceSnapshot(
                id,
                owned ? ProtocolInfo.InstanceModes.Background : ProtocolInfo.InstanceModes.Attach,
                owned,
                year,
                null,
                100,
                false,
                busy,
                documents);

        [Fact]
        public void IdleOtherYearInstanceWithoutDocumentsCanBeRetired()
        {
            Assert.True(InstanceRoom.CanRetire(Background("i1", 2024), 2023));
        }

        [Fact]
        public void InstanceOfThePinnedYearCannotBeRetired()
        {
            Assert.False(InstanceRoom.CanRetire(Background("i1", 2024), 2024));
        }

        [Fact]
        public void BusyInstanceCannotBeRetired()
        {
            Assert.False(InstanceRoom.CanRetire(Background("i1", 2024, busy: true), 2023));
        }

        [Fact]
        public void InstanceHoldingDocumentsCannotBeRetired()
        {
            Assert.False(InstanceRoom.CanRetire(Background("i1", 2024, documents: 1), 2023));
        }

        [Fact]
        public void AttachedInstanceCannotBeRetired()
        {
            Assert.False(InstanceRoom.CanRetire(Background("a1", 2024, owned: false), 2023));
        }

        [Fact]
        public void WithoutAPinNothingIsRetired()
        {
            Assert.False(InstanceRoom.CanRetire(Background("i1", 2024), null));
        }

        [Fact]
        public void UnknownYearCannotBeRetired()
        {
            Assert.False(InstanceRoom.CanRetire(Background("i1", null), 2023));
        }

        [Fact]
        public void FirstEligibleInstanceInListingOrderIsChosen()
        {
            var instances = new[]
            {
                Background("i1", 2024, busy: true),
                Background("i2", 2024),
                Background("i3", 2024),
            };

            Assert.Equal("i2", InstanceRoom.InstanceToRetire(instances, 2023)!.InstanceId);
        }

        [Fact]
        public void NullWhenNothingIsEligible()
        {
            var instances = new[] { Background("i1", 2024, documents: 2), Background("i2", 2023) };

            Assert.Null(InstanceRoom.InstanceToRetire(instances, 2023));
        }
    }

    /// <summary>F6 at the processor: a launch at max_instances closes an idle other-year instance first.</summary>
    public class RoomForLaunchTests
    {
        [Fact]
        public async Task IdleOtherYearInstanceIsRetiredBeforeTheLaunch()
        {
            var h = new Harness();
            h.Manager.Installed.Add(2023);
            h.Manager.AddBackground("i1");

            var response = await h.CallAsync(
                "document.open",
                Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))),
                TestData.Context(version: 2023, maxInstances: 1));

            Assert.True((bool)response["ok"]!);
            Assert.Equal(new[] { "i1" }, h.Manager.Retired);
            Assert.Equal(new[] { "StartNew" }, h.Manager.Targets);
        }

        [Fact]
        public async Task NoRetirementWhileTheLaunchFitsWithinMaxInstances()
        {
            var h = new Harness();
            h.Manager.Installed.Add(2023);
            h.Manager.AddBackground("i1");

            var response = await h.CallAsync(
                "document.open",
                Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))),
                TestData.Context(version: 2023, maxInstances: 2));

            Assert.True((bool)response["ok"]!);
            Assert.Empty(h.Manager.Retired);
        }
    }

    /// <summary>F2: a pin that is not installed is answered before the pool; instance_limit says what is wrong.</summary>
    public class PinNotInstalledAndLimitMessageTests
    {
        [Fact]
        public async Task PinnedYearNotInstalledIsVersionNotInstalledBeforeAnyPoolDecision()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1", documents: 1);

            var response = await h.CallAsync(
                "document.open",
                Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))),
                TestData.Context(version: 2023, maxInstances: 1));

            Assert.Equal("version_not_installed", (string?)response["error"]!["code"]);
            Assert.Equal(2023, (int)response["error"]!["details"]!["pinned_version"]!);
            Assert.Equal(new[] { 2024 }, ((JArray)response["error"]!["details"]!["installed_versions"]!).Select(v => (int)v));
            Assert.Empty(h.Manager.Targets);
        }

        [Fact]
        public async Task PinNotInstalledIsCheckedForDocumentsListToo()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1");

            var response = await h.CallAsync("documents.list", context: TestData.Context(version: 2023));

            Assert.Equal("version_not_installed", (string?)response["error"]!["code"]);
            Assert.Empty(h.Manager.Targets);
        }

        [Fact]
        public async Task BusyPoolKeepsTheBusyMessage()
        {
            var h = new Harness();
            h.Manager.AddBackground("i1", documents: 1, busy: true);

            var response = await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\A.sldprt"))));

            Assert.Equal("instance_limit", (string?)response["error"]!["code"]);
            Assert.Contains("are busy", (string?)response["error"]!["message"]);
        }

        [Fact]
        public async Task FullPoolOfAnotherPinnedYearsDocumentsGivesAnAccurateMessage()
        {
            var h = new Harness();
            h.Manager.Installed.Add(2023);
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\A.sldprt"))));

            var response = await h.CallAsync(
                "document.open",
                Harness.Args(("path", Harness.S(@"C:\Models\B.sldprt"))),
                TestData.Context(version: 2023, maxInstances: 1));

            Assert.Equal("instance_limit", (string?)response["error"]!["code"]);
            Assert.Equal(
                "All 1 background instances are in use (another pinned version or open documents). Close documents or wait for idle shutdown.",
                (string?)response["error"]!["message"]);
            Assert.Single(h.Manager.Instances);
        }
    }

    /// <summary>C6 and C7: always-present list fields, describe texts, and the attach readiness error.</summary>
    public class ContractCorrectionTests
    {
        [Fact]
        public void MateItemAlwaysCarriesErrorCodeAndWarning()
        {
            var json = WireJson.Compact(WireJson.ToToken(new MateItem { Name = "Mate1", Type = "Coincident", Alignment = "Aligned" }));

            Assert.Contains("\"error_code\":\"0\"", json);
            Assert.Contains("\"warning\":false", json);
        }

        [Fact]
        public void FeatureDetailAlwaysCarriesErrorCodeAndWarning()
        {
            var json = WireJson.Compact(WireJson.ToToken(new FeatureDetail { Name = "Boss", Type = "Extrusion", Id = "7" }));

            Assert.Contains("\"error_code\":\"0\"", json);
            Assert.Contains("\"warning\":false", json);
        }

        [Fact]
        public void ErrorAndWarningValuesAreSerializedWhenPresent()
        {
            var item = new FeatureItem { Name = "Boss", Type = "Extrusion", Id = "7", ErrorCode = "1234", Warning = "5" };

            var json = WireJson.Compact(WireJson.ToToken(item));

            Assert.Contains("\"error_code\":\"1234\"", json);
            Assert.Contains("\"warning\":\"5\"", json);
        }

        [Fact]
        public async Task DescribeDocumentsDeclaresReadOnlyAndReferencedDocuments()
        {
            var describe = await new Harness().CallAsync("describe");

            var text = describe.ToString();
            Assert.Contains("read_only reflects how SolidWorks opened the document", text);
            Assert.Contains("Referenced component documents appear with opened_by_host false", text);
        }

        [Fact]
        public async Task DescribeDocumentsTransformLayout()
        {
            var describe = await new Harness().CallAsync("describe");

            var text = describe.ToString();
            Assert.Contains("16 numbers in SolidWorks MathTransform.ArrayData order", text);
            Assert.Contains("[9-11] translation in metres", text);
            Assert.Contains("[12] scale, [13-15] unused", text);
        }

        [Fact]
        public void PropertyResolvedValueIsOmittedWhenNotCached()
        {
            var json = WireJson.Compact(WireJson.ToToken(new PropertyItem { Name = "Mass", Type = "text", Value = "1 kg" }));

            Assert.DoesNotContain("resolved_value", json);
            Assert.Contains("\"value\":\"1 kg\"", json);
        }

        [Fact]
        public void PropertyResolvedValueIsSerializedWhenCached()
        {
            var json = WireJson.Compact(WireJson.ToToken(new PropertyItem { Name = "Mass", Type = "text", Value = "1 kg", ResolvedValue = "1 kg" }));

            Assert.Contains("\"resolved_value\":\"1 kg\"", json);
        }

        [Fact]
        public void AttachNotReadyErrorIsActionable()
        {
            var error = HostError.AttachNotReady("i3", 77);

            Assert.Equal(ErrorCodes.InstanceTimeout, error.Code);
            Assert.Contains("is running but is not answering yet", error.Message);
            Assert.Contains("finish or dismiss the dialog in SolidWorks and try again", error.Message);
            Assert.Equal("i3", (string?)error.Details["instance_id"]);
            Assert.Equal(77, (int)error.Details["pid"]!);
        }
    }
}
