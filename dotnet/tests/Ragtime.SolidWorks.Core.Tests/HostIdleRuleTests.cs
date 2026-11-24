using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class HostIdleRuleTests
    {
        private static InstanceSnapshot Background(string id, int documents = 0, bool busy = false, bool owned = true) =>
            new InstanceSnapshot(id, ProtocolInfo.InstanceModes.Background, owned, 2024, "32.5.0", 100, false, busy, documents);

        private static InstanceSnapshot Attach(string id, int documents = 0, bool busy = false) =>
            new InstanceSnapshot(id, ProtocolInfo.InstanceModes.Attach, false, 2024, "32.5.0", 200, true, busy, documents);

        private static readonly IReadOnlyList<InstanceSnapshot> NoInstances = new List<InstanceSnapshot>();

        public static IEnumerable<object[]> SingleReasonCases()
        {
            yield return new object[] { 1, 100, 0, NoInstances, false, HostIdleRule.RequestInFlight };
            yield return new object[] { 0, 4, 5, NoInstances, false, HostIdleRule.RecentlyUsed };
            yield return new object[] { 0, 100, 0, new[] { Background("i1", busy: true) }, false, HostIdleRule.InstanceBusy };
            yield return new object[] { 0, 100, 0, new[] { Background("i1", documents: 2) }, false, HostIdleRule.DocumentsOpen };
            yield return new object[] { 0, 100, 0, NoInstances, true, HostIdleRule.PdmBusy };
        }

        [Theory]
        [MemberData(nameof(SingleReasonCases))]
        public void EachReasonAloneBlocksStop(int inFlight, int idle, int ifIdle, IReadOnlyList<InstanceSnapshot> instances, bool pdmBusy, string reason)
        {
            var reasons = HostIdleRule.BlockingReasons(inFlight, idle, ifIdle, instances, pdmBusy);

            Assert.Equal(new[] { reason }, reasons);
        }

        [Fact]
        public void CombinedReasonsKeepTheFixedOrder()
        {
            var instances = new[] { Background("i1", documents: 1, busy: true) };

            var reasons = HostIdleRule.BlockingReasons(2, 0, 60, instances, true);

            Assert.Equal(
                new[]
                {
                    HostIdleRule.RequestInFlight,
                    HostIdleRule.RecentlyUsed,
                    HostIdleRule.InstanceBusy,
                    HostIdleRule.DocumentsOpen,
                    HostIdleRule.PdmBusy,
                },
                reasons);
        }

        [Fact]
        public void IdleExactlyAtThresholdIsNotRecentlyUsed()
        {
            Assert.Empty(HostIdleRule.BlockingReasons(0, 60, 60, NoInstances, false));
        }

        [Fact]
        public void ZeroIfIdleNeverBlocksOnRecency()
        {
            Assert.Empty(HostIdleRule.BlockingReasons(0, 0, 0, NoInstances, false));
        }

        [Fact]
        public void AttachModeInstanceWithDocumentsDoesNotBlock()
        {
            var reasons = HostIdleRule.BlockingReasons(0, 100, 0, new[] { Attach("a1", documents: 3) }, false);

            Assert.Empty(reasons);
        }

        [Fact]
        public void NonOwnedBackgroundInstanceWithDocumentsDoesNotBlock()
        {
            var reasons = HostIdleRule.BlockingReasons(0, 100, 0, new[] { Background("i1", documents: 3, owned: false) }, false);

            Assert.Empty(reasons);
        }

        [Fact]
        public void BusyAttachInstanceCountsAsInstanceBusy()
        {
            var reasons = HostIdleRule.BlockingReasons(0, 100, 0, new[] { Attach("a1", busy: true) }, false);

            Assert.Equal(new[] { HostIdleRule.InstanceBusy }, reasons);
        }

        [Fact]
        public void EmptyListMeansSafeToStop()
        {
            Assert.Empty(HostIdleRule.BlockingReasons(0, 500, 60, NoInstances, false));
        }

        [Fact]
        public void ActivityInfoSerializesSnakeCaseFields()
        {
            var status = new StatusResult { Activity = new HostActivityInfo { InFlight = 2, IdleSeconds = 7, PdmBusy = true } };

            var activity = (JObject)WireJson.ToToken(status)["activity"]!;

            Assert.Equal(2, (int)activity["in_flight"]!);
            Assert.Equal(7, (int)activity["idle_seconds"]!);
            Assert.True((bool)activity["pdm_busy"]!);
        }

        [Fact]
        public void StatusActivityDefaultsToIdleEmptyFacts()
        {
            var activity = new StatusResult().Activity;

            Assert.Equal(0, activity.InFlight);
            Assert.Equal(0, activity.IdleSeconds);
            Assert.False(activity.PdmBusy);
        }
    }
}
