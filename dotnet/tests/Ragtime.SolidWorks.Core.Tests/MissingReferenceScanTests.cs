using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>The document.open missing-reference scan (review finding 6): the 50-path cap and the deadline between probes.</summary>
    public class MissingReferenceScanTests
    {
        [Fact]
        public void ProbingStopsAfterTheMissingLimitAndNeverChecksTheRest()
        {
            var references = Enumerable.Range(0, 500).Select(i => @"C:\Parts\Ref" + i + ".sldprt").ToList();
            var probes = 0;
            var clock = new FakeClock();

            var missing = MissingReferenceScan.Collect(
                references,
                path => { probes++; return false; },
                new Deadline(60, clock),
                DocumentOpenResult.MaxMissingReferences);

            Assert.Equal(DocumentOpenResult.MaxMissingReferences, missing.Count);
            Assert.Equal(50, probes);
            Assert.Equal(references.Take(50), missing);
        }

        [Fact]
        public void PresentReferencesAreNotListed()
        {
            var references = new List<string> { @"C:\a.sldprt", @"C:\b.sldprt", @"C:\c.sldprt" };

            var missing = MissingReferenceScan.Collect(
                references,
                path => path != @"C:\b.sldprt",
                new Deadline(60, new FakeClock()),
                50);

            Assert.Equal(new[] { @"C:\b.sldprt" }, missing);
        }

        [Fact]
        public void AnExpiredDeadlineStopsBeforeTheFirstProbe()
        {
            var clock = new FakeClock();
            var deadline = new Deadline(1, clock);
            clock.Advance(TimeSpan.FromSeconds(2));
            var probes = 0;

            var missing = MissingReferenceScan.Collect(
                new[] { @"C:\a.sldprt", @"C:\b.sldprt" },
                path => { probes++; return false; },
                deadline,
                50);

            Assert.Empty(missing);
            Assert.Equal(0, probes);
        }

        [Fact]
        public void TheDeadlineIsCheckedBetweenProbes()
        {
            var clock = new FakeClock();
            var deadline = new Deadline(1, clock);
            var probes = 0;
            var references = Enumerable.Range(0, 10).Select(i => @"C:\Ref" + i + ".sldprt");

            var missing = MissingReferenceScan.Collect(
                references,
                path =>
                {
                    probes++;
                    clock.Advance(TimeSpan.FromMilliseconds(400));
                    return false;
                },
                deadline,
                50);

            // Three probes fit in one second (0.4 s each); the fourth starts after the deadline has passed.
            Assert.Equal(3, probes);
            Assert.Equal(3, missing.Count);
        }

        [Fact]
        public void TheMissingListCannotExceedTheDocumentOpenCap()
        {
            Assert.Equal(50, DocumentOpenResult.MaxMissingReferences);
        }
    }
}
