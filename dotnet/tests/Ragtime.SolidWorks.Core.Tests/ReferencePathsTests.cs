using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ReferencePathsTests
    {
        [Theory]
        [InlineData(@"C:\PDM\Parts", "A.SLDPRT", @"\Parts\A.SLDPRT")]
        [InlineData(@"C:\PDM\Parts\Deep", "A.SLDASM", @"\Parts\Deep\A.SLDASM")]
        [InlineData(@"C:\PDM", "A.SLDPRT", @"\A.SLDPRT")]
        [InlineData(@"C:\PDM\Sub\..\Other", "A.SLDPRT", @"\Other\A.SLDPRT")]
        [InlineData(@"C:\PDM\\Parts\.\X", "A.SLDPRT", @"\Parts\X\A.SLDPRT")]
        [InlineData("C:/PDM/Parts", "A.SLDPRT", @"\Parts\A.SLDPRT")]
        [InlineData(@"c:\pdm\Parts", "A.SLDPRT", @"\Parts\A.SLDPRT")]
        [InlineData(@"C:\PDM\Parts", "Copy of 2 position^32-010.SLDPRT", @"\Parts\Copy of 2 position^32-010.SLDPRT")]
        public void ReturnsTheVaultPathInsideTheRoot(string folder, string name, string expected) =>
            Assert.Equal(expected, ReferencePaths.VaultPath(@"C:\PDM", folder, name));

        [Fact]
        public void TrailingSeparatorOnRootChangesNothing()
        {
            Assert.Equal(@"\Parts\A.SLDPRT", ReferencePaths.VaultPath(@"C:\PDM\", @"C:\PDM\Parts", "A.SLDPRT"));
            Assert.Equal(@"\A.SLDPRT", ReferencePaths.VaultPath(@"C:\PDM\", @"C:\PDM", "A.SLDPRT"));
        }

        [Theory]
        [InlineData(@"C:\PDM2\x")]
        [InlineData(@"C:\PDM2")]
        [InlineData(@"C:\PDMX\Parts")]
        [InlineData(@"D:\PDM\Parts")]
        [InlineData(@"C:\Other")]
        [InlineData(@"C:\PDM\..\Other")]
        [InlineData(@"C:\PDM\..\..\..\x")]
        [InlineData(@"C:\")]
        [InlineData(@"\\server\share\PDM\Parts")]
        [InlineData(@"PDM\Parts")]
        [InlineData(@"C:PDM\Parts")]
        public void RejectsFoldersOutsideTheRootOrNotAbsolute(string folder) =>
            Assert.Null(ReferencePaths.VaultPath(@"C:\PDM", folder, "A.SLDPRT"));

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        public void RejectsAMissingFolder(string? folder) =>
            Assert.Null(ReferencePaths.VaultPath(@"C:\PDM", folder, "A.SLDPRT"));

        [Theory]
        [InlineData("A<B.SLDPRT")]
        [InlineData("A>B.SLDPRT")]
        [InlineData("A|B.SLDPRT")]
        [InlineData("A\"B.SLDPRT")]
        [InlineData("A:B.SLDPRT")]
        [InlineData("A*B.SLDPRT")]
        [InlineData("A?B.SLDPRT")]
        [InlineData(@"A\B.SLDPRT")]
        [InlineData("A/B.SLDPRT")]
        [InlineData("A\u0001B.SLDPRT")]
        [InlineData("SW-54784-A<1>")]
        [InlineData(".")]
        [InlineData("..")]
        [InlineData("Name.")]
        [InlineData("Name ")]
        [InlineData("")]
        [InlineData(null)]
        public void RejectsNamesThatAreNotWindowsFileNames(string? name) =>
            Assert.Null(ReferencePaths.VaultPath(@"C:\PDM", @"C:\PDM\Parts", name));

    }
}
