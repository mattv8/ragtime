using System;
using System.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>The copy size and quota rules of document.copy (review finding 7), with fake sizes and free space.</summary>
    public class CopyQuotaTests
    {
        private const long Gib = 1024L * 1024 * 1024;

        [Fact]
        public void TheLimitsAreTwoTensAndFiveGibibytes()
        {
            Assert.Equal(2 * Gib, DocumentCopyLimits.MaxCopyBytes);
            Assert.Equal(10 * Gib, DocumentCopyLimits.MaxCopiesFolderBytes);
            Assert.Equal(5 * Gib, DocumentCopyLimits.MinFreeBytesAfterCopy);
        }

        [Fact]
        public void AdditionSaturatesInsteadOfWrapping()
        {
            Assert.Equal(long.MaxValue, CopyQuota.Add(long.MaxValue, 1));
            Assert.Equal(long.MaxValue, CopyQuota.Add(long.MaxValue - 1, long.MaxValue));
            Assert.Equal(3, CopyQuota.Add(1, 2));
        }

        [Fact]
        public void NegativeSizesAreRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => CopyQuota.Add(-1, 2));
        }

        [Fact]
        public void SizesSumWithoutOverflowEvenForHugeFiles()
        {
            var sizes = Enumerable.Repeat(long.MaxValue / 2, 5);

            Assert.Equal(long.MaxValue, CopyQuota.SumSizes(sizes));
        }

        [Fact]
        public void ACopyOfExactlyTheSizeLimitPasses()
        {
            CopyQuota.RequireWithinLimits(2 * Gib, 0, 100 * Gib);
        }

        [Fact]
        public void ACopyOneByteOverTheSizeLimitIsRefusedWithCopySizeDetails()
        {
            var error = Assert.Throws<HostErrorException>(() => CopyQuota.RequireWithinLimits(2 * Gib + 1, 0, 100 * Gib));

            Assert.Equal(ErrorCodes.ResultTooLarge, error.Code);
            Assert.Equal("copy_size", (string)error.Details["reason"]!);
            Assert.Equal(2 * Gib, (long)error.Details["limit_bytes"]!);
            Assert.Equal(2 * Gib + 1, (long)error.Details["size_bytes"]!);
        }

        [Fact]
        public void ACopiesFolderOfExactlyTheQuotaPasses()
        {
            CopyQuota.RequireWithinLimits(1, 10 * Gib, 100 * Gib);
        }

        [Fact]
        public void ACopiesFolderOverTheQuotaIsRefusedWithCopiesQuotaDetails()
        {
            var error = Assert.Throws<HostErrorException>(() => CopyQuota.RequireWithinLimits(1, 10 * Gib + 1, 100 * Gib));

            Assert.Equal(ErrorCodes.ResultTooLarge, error.Code);
            Assert.Equal("copies_quota", (string)error.Details["reason"]!);
            Assert.Equal(10 * Gib, (long)error.Details["limit_bytes"]!);
            Assert.Equal(10 * Gib + 1, (long)error.Details["size_bytes"]!);
        }

        [Fact]
        public void TheVolumeKeepingExactlyTheMinimumFreeSpacePasses()
        {
            CopyQuota.RequireWithinLimits(1 * Gib, 0, 1 * Gib + 5 * Gib);
        }

        [Fact]
        public void TheVolumeKeepingLessThanTheMinimumFreeSpaceIsRefused()
        {
            var error = Assert.Throws<HostErrorException>(() => CopyQuota.RequireWithinLimits(1 * Gib, 0, 1 * Gib + 5 * Gib - 1));

            Assert.Equal(ErrorCodes.ResultTooLarge, error.Code);
            Assert.Equal("copies_quota", (string)error.Details["reason"]!);
            Assert.Equal(5 * Gib, (long)error.Details["limit_bytes"]!);
            Assert.Equal(1 * Gib, (long)error.Details["size_bytes"]!);
            Assert.Equal(1 * Gib + 5 * Gib - 1, (long)error.Details["free_bytes"]!);
        }

        [Fact]
        public void ASizeLimitIsCheckedBeforeTheQuota()
        {
            var error = Assert.Throws<HostErrorException>(() => CopyQuota.RequireWithinLimits(3 * Gib, 11 * Gib, 0));

            Assert.Equal("copy_size", (string)error.Details["reason"]!);
        }
    }
}
