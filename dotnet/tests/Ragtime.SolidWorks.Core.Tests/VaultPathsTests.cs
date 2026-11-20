using System;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class VaultPathsTests
    {
        [Theory]
        [InlineData(@"\A\B", null, @"\A\B")]
        [InlineData("/A/B", null, @"\A\B")]
        [InlineData(@"\", null, @"\")]
        [InlineData(@"c:\pdm\A", @"C:\PDM", @"\A")]
        [InlineData(@"c:\pdm", @"C:\PDM", @"\")]
        [InlineData(@"\A\", null, @"\A")]
        public void Normalizes(string input, string? root, string expected) =>
            Assert.Equal(expected, VaultPaths.Normalize(input, root, "path"));

        [Theory]
        [InlineData(@"C:\PDM2\x", @"C:\PDM")]
        [InlineData(@"D:\PDM\x", @"C:\PDM")]
        [InlineData(@"\\server\share", null)]
        [InlineData(@"\\?\C:\PDM", null)]
        [InlineData("", null)]
        [InlineData("\\A\u007f", null)]
        [InlineData(@"\A:B", null)]
        [InlineData(@"\A*B", null)]
        [InlineData(@"\A?B", null)]
        [InlineData("\\A\"B", null)]
        [InlineData(@"\A<B", null)]
        [InlineData(@"\A>B", null)]
        [InlineData(@"\A|B", null)]
        [InlineData(@"\.", null)]
        [InlineData(@"\..", null)]
        [InlineData(@"\A\\B", null)]
        [InlineData(@"\ A", null)]
        [InlineData(@"\A ", null)]
        [InlineData(@"\A.", null)]
        public void RejectsInvalidPaths(string input, string? root)
        {
            var error = Assert.Throws<HostErrorException>(() => VaultPaths.Normalize(input, root, "folder"));
            Assert.Equal(ErrorCodes.InvalidArguments, error.Code);
            Assert.Equal("folder", (string?)error.Details["field"]);
        }

        [Fact]
        public void RejectsLongAndUnknownAbsolutePaths()
        {
            Assert.Throws<HostErrorException>(() => VaultPaths.Normalize("\\" + new string('a', 260), null, "path"));
            var error = Assert.Throws<HostErrorException>(() => VaultPaths.Normalize(@"C:\PDM\A", null, "path"));
            Assert.Equal("Give the path relative to the vault, starting with a backslash.", error.Message);
        }

        [Theory]
        [InlineData(@"\13_Ragtime", true)]
        [InlineData(@"\13_ragtime\A", true)]
        [InlineData(@"\13_Ragtime2", false)]
        [InlineData(@"\", false)]
        public void ChecksApproval(string path, bool expected) =>
            Assert.Equal(expected, VaultPaths.IsApproved(path, new[] { @"\13_Ragtime" }));

        [Fact]
        public void PathHelpers()
        {
            Assert.False(VaultPaths.IsApproved(@"\A", Array.Empty<string>()));
            Assert.Equal(@"\A", VaultPaths.Parent(@"\A\B"));
            Assert.Equal(@"\", VaultPaths.Parent(@"\A"));
            Assert.Equal(@"\A", VaultPaths.Combine(@"\", "A"));
            Assert.Equal(@"\A\B", VaultPaths.Combine(@"\A", "B"));
            Assert.Null(VaultPaths.FindNameProblem("part.txt"));
            foreach (var name in new[] { "", ".", "..", " A", "A ", "A.", "A/B", "A\\B", "A:B", "A*B", new string('x', 256) })
                Assert.NotNull(VaultPaths.FindNameProblem(name));
        }

        [Theory]
        [InlineData("A?B")]
        [InlineData("A\"B")]
        [InlineData("A<B")]
        [InlineData("A>B")]
        [InlineData("A|B")]
        [InlineData("A\u007f")]
        public void RejectsInvalidNames(string name) => Assert.NotNull(VaultPaths.FindNameProblem(name));

        [Theory]
        [InlineData(@"C:\Temp\a.txt", true)]
        [InlineData(@"\\server\share\a.txt", false)]
        [InlineData(@"C:\Temp\..\a.txt", false)]
        public void ChecksLocalFilePath(string path, bool valid) =>
            Assert.Equal(valid, DocumentPaths.FindLocalFileProblem(path) == null);

        [Theory]
        [InlineData("path")]
        [InlineData("source")]
        public void LocalFileCheckUsesRequestedField(string field)
        {
            var error = Assert.Throws<HostErrorException>(() => LocalFileCheck.Require(@"\\server\share\file", field));
            Assert.Equal(field, (string?)error.Details["field"]);
        }
    }
}
