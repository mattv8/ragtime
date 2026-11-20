using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class InstancePoolTests
    {
        private static InstanceSnapshot Background(string id, int documents = 0, bool busy = false, int year = 2024) =>
            new InstanceSnapshot(id, "background", true, year, year + ".0", 100, false, busy, documents);

        [Fact]
        public void NoInstanceStartsOne()
        {
            var choice = InstancePool.ChooseForOpen(new List<InstanceSnapshot>(), 1);

            Assert.Equal(PoolAction.StartNew, choice.Action);
            Assert.Null(choice.InstanceId);
        }

        [Fact]
        public void IdleInstanceIsUsedEvenWhenPoolIsNotFull()
        {
            var choice = InstancePool.ChooseForOpen(new[] { Background("i1") }, 3);

            Assert.Equal(PoolAction.UseExisting, choice.Action);
            Assert.Equal("i1", choice.InstanceId);
        }

        [Fact]
        public void LoadedInstanceStartsSecondWhenRoomRemains()
        {
            var choice = InstancePool.ChooseForOpen(new[] { Background("i1", documents: 2) }, 2);

            Assert.Equal(PoolAction.StartNew, choice.Action);
        }

        [Fact]
        public void LoadedInstanceIsReusedWhenPoolIsFull()
        {
            var choice = InstancePool.ChooseForOpen(new[] { Background("i1", documents: 2) }, 1);

            Assert.Equal(PoolAction.UseExisting, choice.Action);
            Assert.Equal("i1", choice.InstanceId);
        }

        [Fact]
        public void LeastLoadedIsChosenWhenPoolIsFull()
        {
            var instances = new[] { Background("i1", documents: 2), Background("i2", documents: 1) };

            var choice = InstancePool.ChooseForOpen(instances, 2);

            Assert.Equal(PoolAction.UseExisting, choice.Action);
            Assert.Equal("i2", choice.InstanceId);
        }

        [Fact]
        public void FullPoolWithAllBusyInstancesIsInstanceLimit()
        {
            var instances = new[] { Background("i1", documents: 1, busy: true), Background("i2", busy: true) };

            var choice = InstancePool.ChooseForOpen(instances, 2);

            Assert.Equal(PoolAction.LimitReached, choice.Action);
            Assert.Null(choice.InstanceId);
        }

        [Fact]
        public void BusyIdleInstanceStillLeavesRoomForANewOne()
        {
            var choice = InstancePool.ChooseForOpen(new[] { Background("i1", busy: true) }, 2);

            Assert.Equal(PoolAction.StartNew, choice.Action);
        }

        [Fact]
        public void BusyInstanceWithNoRoomIsLimitReached()
        {
            var choice = InstancePool.ChooseForOpen(new[] { Background("i1", busy: true) }, 1);

            Assert.Equal(PoolAction.LimitReached, choice.Action);
        }

        [Fact]
        public void NonBusyInstanceBeatsBusyOneEvenWithDocuments()
        {
            var instances = new[] { Background("i1", documents: 0, busy: true), Background("i2", documents: 3) };

            var choice = InstancePool.ChooseForOpen(instances, 1);

            Assert.Equal(PoolAction.UseExisting, choice.Action);
            Assert.Equal("i2", choice.InstanceId);
        }

        [Fact]
        public void AttachInstancesAreIgnoredByThePool()
        {
            var attached = new InstanceSnapshot("a1", "attach", false, 2024, "32.5.0", 50, true, false, 0);

            var choice = InstancePool.ChooseForOpen(new[] { attached }, 1);

            Assert.Equal(PoolAction.StartNew, choice.Action);
        }

        [Fact]
        public void MaxInstancesBelowOneIsRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => InstancePool.ChooseForOpen(new List<InstanceSnapshot>(), 0));
        }
    }

    public class AttachSelectionTests
    {
        [Fact]
        public void NoCandidatesIsSolidworksNotRunning()
        {
            var error = Assert.Throws<HostErrorException>(() => AttachSelection.Choose(new List<AttachCandidate>(), 2024));

            Assert.Equal(ErrorCodes.SolidworksNotRunning, error.Code);
        }

        [Fact]
        public void PinnedYearWithOnlyOtherYearsIsVersionMismatch()
        {
            var candidates = new[] { new AttachCandidate(10, 2025) };

            var error = Assert.Throws<HostErrorException>(() => AttachSelection.Choose(candidates, 2024));

            Assert.Equal(ErrorCodes.VersionMismatch, error.Code);
            Assert.Equal(2024, (int)error.Details["pinned_version"]!);
        }

        [Fact]
        public void PinnedYearPicksLowestPidAmongMatches()
        {
            var candidates = new[]
            {
                new AttachCandidate(30, 2024),
                new AttachCandidate(20, 2024),
                new AttachCandidate(5, 2025),
            };

            Assert.Equal(20, AttachSelection.Choose(candidates, 2024).ProcessId);
        }

        [Fact]
        public void NoPinPicksLowestPidOverall()
        {
            var candidates = new[] { new AttachCandidate(9, 2025), new AttachCandidate(3, 2024) };

            Assert.Equal(3, AttachSelection.Choose(candidates, null).ProcessId);
        }

        [Fact]
        public void UnknownYearDoesNotMatchAPin()
        {
            var candidates = new[] { new AttachCandidate(7, null) };

            Assert.Equal(ErrorCodes.VersionMismatch, Assert.Throws<HostErrorException>(() => AttachSelection.Choose(candidates, 2024)).Code);
            Assert.Equal(7, AttachSelection.Choose(candidates, null).ProcessId);
        }
    }

    public class IdlePolicyTests
    {
        [Theory]
        [InlineData("background", true, false, 900, 900, true)]
        [InlineData("background", true, false, 899, 900, false)]
        [InlineData("background", true, true, 1000, 900, false)]
        [InlineData("background", false, false, 1000, 900, false)]
        [InlineData("attach", true, false, 1000, 900, false)]
        public void EligibilityTable(string mode, bool owned, bool busy, int idleSeconds, int limit, bool expected)
        {
            Assert.Equal(expected, IdlePolicy.IsEligibleForIdleShutdown(mode, owned, busy, TimeSpan.FromSeconds(idleSeconds), limit));
        }
    }

    public class DocumentRegistryTests
    {
        private static SessionDocumentInfo Doc(string path, string title = "") =>
            new SessionDocumentInfo(
                title.Length > 0 ? title : path.Substring(path.LastIndexOf('\\') + 1),
                path,
                "part",
                "Default",
                true,
                false);

        [Fact]
        public void IdsAreSequentialAndNeverReused()
        {
            var registry = new DocumentRegistry();

            var first = registry.Register("i1", Doc(@"C:\a\one.sldprt"), true);
            var second = registry.Register("i1", Doc(@"C:\a\two.sldprt"), true);
            registry.Remove(first.DocumentId);
            var third = registry.Register("i1", Doc(@"C:\a\three.sldprt"), true);

            Assert.Equal("doc-1", first.DocumentId);
            Assert.Equal("doc-2", second.DocumentId);
            Assert.Equal("doc-3", third.DocumentId);
        }

        [Fact]
        public void FindByIdAndByPathCaseInsensitive()
        {
            var registry = new DocumentRegistry();
            var entry = registry.Register("i1", Doc(@"C:\Models\Bracket.sldprt"), true);

            Assert.Same(entry, registry.Find("doc-1"));
            Assert.Same(entry, registry.Find(@"c:\models\bracket.SLDPRT"));
            Assert.Null(registry.Find("doc-9"));
            Assert.Null(registry.Find(@"C:\Models\Other.sldprt"));
            Assert.Null(registry.Find(null));
        }

        [Fact]
        public void DocumentIdShapeIsStrict()
        {
            Assert.True(DocumentRegistry.LooksLikeDocumentId("doc-1"));
            Assert.True(DocumentRegistry.LooksLikeDocumentId("doc-12"));
            Assert.False(DocumentRegistry.LooksLikeDocumentId("doc-0"));
            Assert.False(DocumentRegistry.LooksLikeDocumentId("doc-01"));
            Assert.False(DocumentRegistry.LooksLikeDocumentId("doc-"));
            Assert.False(DocumentRegistry.LooksLikeDocumentId(@"C:\x.sldprt"));
        }

        [Fact]
        public void RegisteringSameDocumentKeepsIdAndMergesOpenedByHost()
        {
            var registry = new DocumentRegistry();
            var first = registry.Register("i1", Doc(@"C:\a\p.sldprt"), false);

            var second = registry.Register("i1", Doc(@"C:\A\P.sldprt"), true);

            Assert.Equal(first.DocumentId, second.DocumentId);
            Assert.True(second.OpenedByHost);
            Assert.Single(registry.Snapshot());
        }

        [Fact]
        public void SameFileInTwoInstancesGetsTwoIds()
        {
            var registry = new DocumentRegistry();

            var a = registry.Register("i1", Doc(@"C:\a\p.sldprt"), true);
            var b = registry.Register("i2", Doc(@"C:\a\p.sldprt"), true);

            Assert.NotEqual(a.DocumentId, b.DocumentId);
            Assert.Equal(2, registry.Snapshot().Count);
        }

        [Fact]
        public void SynchronizeKeepsListedDocumentsDropsMissingAndAddsNew()
        {
            var registry = new DocumentRegistry();
            var kept = registry.Register("i1", Doc(@"C:\a\keep.sldprt"), true);
            registry.Register("i1", Doc(@"C:\a\gone.sldprt"), true);
            registry.Register("i2", Doc(@"C:\b\other.sldprt"), true);

            registry.Synchronize("i1", new[] { Doc(@"C:\a\keep.sldprt"), Doc(@"C:\a\new.sldprt") });

            var all = registry.Snapshot();
            Assert.Equal(3, all.Count);
            var keptNow = all.Single(e => e.Path.EndsWith("keep.sldprt", StringComparison.Ordinal));
            Assert.Equal(kept.DocumentId, keptNow.DocumentId);
            Assert.True(keptNow.OpenedByHost);
            var added = all.Single(e => e.Path.EndsWith("new.sldprt", StringComparison.Ordinal));
            Assert.False(added.OpenedByHost);
            Assert.Null(registry.Find(@"C:\a\gone.sldprt"));
            Assert.Equal(1, registry.CountForInstance("i2"));
        }

        [Fact]
        public void ForgetInstanceRemovesOnlyThatInstance()
        {
            var registry = new DocumentRegistry();
            registry.Register("i1", Doc(@"C:\a\one.sldprt"), true);
            registry.Register("i2", Doc(@"C:\a\two.sldprt"), true);

            registry.ForgetInstance("i1");

            Assert.Equal(0, registry.CountForInstance("i1"));
            Assert.Equal(1, registry.CountForInstance("i2"));
        }

        [Fact]
        public void RetainInstancesForgetsDeadOnes()
        {
            var registry = new DocumentRegistry();
            registry.Register("i1", Doc(@"C:\a\one.sldprt"), true);
            registry.Register("i2", Doc(@"C:\a\two.sldprt"), true);

            registry.RetainInstances(new[] { "i2" });

            Assert.Single(registry.Snapshot());
            Assert.Equal("i2", registry.Snapshot()[0].InstanceId);
        }

        [Fact]
        public void TitleMatchesWhenPathIsEmpty()
        {
            var registry = new DocumentRegistry();
            var entry = registry.Register("i1", new SessionDocumentInfo("Untitled", "", "part", null, true, false), false);

            registry.Synchronize("i1", new[] { new SessionDocumentInfo("untitled", "", "part", null, true, false) });

            Assert.Equal(entry.DocumentId, registry.Snapshot()[0].DocumentId);
        }

        [Fact]
        public void ToItemUsesWireNames()
        {
            var registry = new DocumentRegistry();
            var entry = registry.Register("i1", Doc(@"C:\a\p.sldprt"), true);

            var json = WireJson.Compact(WireJson.ToToken(entry.ToItem()));

            Assert.Contains("\"document_id\":\"doc-1\"", json);
            Assert.Contains("\"instance_id\":\"i1\"", json);
            Assert.Contains("\"opened_by_host\":true", json);
            Assert.Contains("\"read_only\":true", json);
            Assert.DoesNotContain("\n", json);
        }
    }

    public class DtoWireNameTests
    {
        [Fact]
        public void SketchSummaryUsesIs3DName()
        {
            var json = WireJson.Compact(WireJson.ToToken(new SketchSummary { Name = "Sketch1", Is3D = true, Units = "SI: m, kg, s, rad" }));

            Assert.Contains("\"is_3d\":true", json);
            Assert.Contains("\"constrained_status\":\"\"", json);
        }

        [Fact]
        public void NullOptionalFieldsAreOmittedButErrorCodeAndWarningAlwaysPresent()
        {
            var json = WireJson.Compact(WireJson.ToToken(new FeatureItem { Name = "Boss", Type = "Extrusion", Id = "7" }));

            Assert.Contains("\"error_code\":\"0\"", json);
            Assert.Contains("\"warning\":false", json);
            Assert.DoesNotContain("parent_names", json);
            Assert.Contains("\"dimension_count\":0", json);
            Assert.Contains("\"suppressed\":false", json);
        }

        [Fact]
        public void ModelSummaryOmitsPartOnlyFieldsForAssemblies()
        {
            var json = WireJson.Compact(WireJson.ToToken(new ModelSummary { Type = "assembly", ComponentCount = 4, MateCount = 2 }));

            Assert.Contains("\"component_count\":4", json);
            Assert.Contains("\"mate_count\":2", json);
            Assert.DoesNotContain("solid_body_count", json);
        }

        [Fact]
        public void EveryListItemTypeSerializesSnakeCase()
        {
            var json = WireJson.Compact(WireJson.ToToken(new ComponentItem { Name = "Part1-1", IsVirtual = true, ChildrenCount = 0, Transform = new double[16] }));

            Assert.Contains("\"is_virtual\":true", json);
            Assert.Contains("\"children_count\":0", json);
            Assert.Contains("\"transform\":[", json);
        }

        [Fact]
        public void StatusInstanceUsesPidAndVersionNames()
        {
            var info = new InstanceInfo { InstanceId = "i1", Mode = "background", Owned = true, Version = 2024, Pid = 4242, Documents = 1 };

            var root = JObject.Parse(WireJson.Compact(WireJson.ToToken(info)));

            Assert.Equal(4242, (int)root["pid"]!);
            Assert.Equal(2024, (int)root["version"]!);
            Assert.Equal("i1", (string?)root["instance_id"]);
            Assert.True(root.ContainsKey("revision"));
            Assert.Equal(JTokenType.Null, root["revision"]!.Type);
        }

        [Fact]
        public void ResultsContainNoStrayPascalCaseNames()
        {
            var json = WireJson.Compact(WireJson.ToToken(new MassPropertiesResult { CenterOfMass = new double[3] }));

            Assert.Contains("\"center_of_mass\"", json);
            Assert.Contains("\"surface_area\"", json);
            Assert.DoesNotContain("CenterOfMass", json);
        }
    }
}
