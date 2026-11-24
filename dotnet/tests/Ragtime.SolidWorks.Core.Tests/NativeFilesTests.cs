using System;
using System.Collections.Generic;
using System.IO;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public sealed class NativeFilesTests
    {
        private const string Root = @"C:\Models";

        [Fact]
        public void SharedReadWorksWhileFileIsOpenForWriting()
        {
            using (var folder = new TempWorkFolder())
            {
                var path = Path.Combine(folder.Path, "part.sldprt");
                File.WriteAllText(path, "native contents");
                using (var writer = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite))
                using (var reader = new StreamReader(NativeFiles.OpenSharedRead(path)))
                {
                    Assert.Equal("native contents", reader.ReadToEnd());
                }
            }
        }

        [Theory]
        [InlineData("assembly", "zip")]
        [InlineData("drawing", "slddrw")]
        [InlineData("part", "sldprt")]
        [InlineData("unknown", "sldprt")]
        public void NativeExtensionFollowsTheDocumentType(string type, string extension)
        {
            Assert.Equal(extension, NativeFiles.Extension(type));
        }

        [Theory]
        [InlineData(@"C:\Models\Parts\Bracket.SLDPRT", true)]
        [InlineData(@"C:\Models\Frame.sldasm", true)]
        [InlineData(@"C:\Models\Sheet.slddrw", true)]
        [InlineData(@"C:\Models\Bracket.step", false)]
        [InlineData(@"C:\Models\Bracket.sld", false)]
        [InlineData("Bracket.sldprt.txt", false)]
        [InlineData("", false)]
        public void OnlySolidWorksFilesAreReferences(string path, bool expected)
        {
            Assert.Equal(expected, NativeFiles.IsSolidWorksFile(path));
        }

        [Fact]
        public void TheSizeCeilingIsExactlyOneHundredMegabytes()
        {
            Assert.Equal(100L * 1024 * 1024, NativeFiles.MaxArchiveBytes);
            NativeFiles.RequireWithinLimit(NativeFiles.MaxArchiveBytes);
        }

        [Fact]
        public void OneByteOverTheCeilingIsExportFailed()
        {
            var error = Assert.Throws<HostErrorException>(() => NativeFiles.RequireWithinLimit(NativeFiles.MaxArchiveBytes + 1));

            Assert.Equal(ErrorCodes.ExportFailed, error.Code);
        }

        [Fact]
        public void NestedAssemblyReturnsEveryLevelOfTheTraversal()
        {
            // What GetDocumentDependencies2 returns with traversal on: a flat list of name/path pairs for the top
            // assembly's children, its sub-assembly, and the sub-assembly's parts. Bracket appears under both.
            var pairs = new List<string>
            {
                "Sub.sldasm", @"C:\Models\Sub\Sub.sldasm",
                "Bracket.sldprt", @"C:\Models\Sub\Parts\Bracket.sldprt",
                "Bolt.sldprt", @"C:\Models\Lib\Bolt.sldprt",
                "Bracket.sldprt", @"C:\Models\Sub\Parts\Bracket.sldprt",
            };

            var paths = NativeFiles.PathsFromDependencyPairs(pairs);

            Assert.Equal(
                new[] { @"C:\Models\Sub\Sub.sldasm", @"C:\Models\Sub\Parts\Bracket.sldprt", @"C:\Models\Lib\Bolt.sldprt" },
                paths);
            Assert.Equal("Sub/Parts/Bracket.sldprt", NativeFiles.EntryNameFor(@"C:\Models", paths[1]));
            Assert.Equal("Sub/Sub.sldasm", NativeFiles.EntryNameFor(@"C:\Models", paths[0]));
        }

        [Fact]
        public void PairParsingIgnoresNamesAndEntriesThatAreNotPaths()
        {
            var pairs = new List<string> { "Bolt.sldprt", @"C:\Lib\Bolt.sldprt", "Orphan.sldprt", "NotAPath.sldprt" };

            var paths = NativeFiles.PathsFromDependencyPairs(pairs);

            Assert.Equal(new[] { @"C:\Lib\Bolt.sldprt" }, paths);
            Assert.Empty(NativeFiles.PathsFromDependencyPairs(new List<string>()));
        }

        [Fact]
        public void ReferenceInsideTheAssemblyFolderKeepsItsRelativeStructure()
        {
            Assert.Equal("Parts/Bracket.sldprt", NativeFiles.EntryNameFor(Root, @"C:\Models\Parts\Bracket.sldprt"));
        }

        [Fact]
        public void ReferenceOutsideTheFolderIsStoredByFileName()
        {
            Assert.Equal("Bolt.sldprt", NativeFiles.EntryNameFor(Root, @"D:\Library\Bolt.sldprt"));
        }

        [Fact]
        public void DotDotSegmentsFallBackToTheFileName()
        {
            Assert.Equal("Secret.sldprt", NativeFiles.EntryNameFor(Root, @"C:\Models\..\Secret\Secret.sldprt"));
            Assert.False(NativeFiles.IsSafeEntryName("Parts/../../Secret.sldprt"));
            Assert.False(NativeFiles.IsSafeEntryName(".."));
        }

        [Fact]
        public void UncPathsFallBackToTheFileNameAndAreNotSafeEntries()
        {
            Assert.Equal("Shared.sldprt", NativeFiles.EntryNameFor(Root, @"\\server\share\Shared.sldprt"));
            Assert.False(NativeFiles.IsSafeEntryName(@"\\server\share\x.sldprt"));
            Assert.False(NativeFiles.IsSafeEntryName("//server/share/x.sldprt"));
        }

        [Theory]
        [InlineData("D:/Library/Bolt.sldprt")]
        [InlineData("C:x.sldprt")]
        [InlineData("Parts/C:evil.sldprt")]
        public void DriveQualifiedNamesAreNotSafeEntries(string entry)
        {
            Assert.False(NativeFiles.IsSafeEntryName(entry));
        }

        [Theory]
        [InlineData("/etc/passwd")]
        [InlineData(@"\windows\x.sldprt")]
        [InlineData("Parts//Bracket.sldprt")]
        [InlineData("Parts\\Bracket.sldprt")]
        [InlineData("./Bracket.sldprt")]
        [InlineData("")]
        public void LeadingSlashesEmptySegmentsAndBackslashesAreNotSafe(string entry)
        {
            Assert.False(NativeFiles.IsSafeEntryName(entry));
        }

        [Fact]
        public void SafeRelativeNamesAreAccepted()
        {
            Assert.True(NativeFiles.IsSafeEntryName("Parts/Sub Folder/Bracket (copy).sldprt"));
            Assert.True(NativeFiles.IsSafeEntryName("Bracket.sldprt"));
        }

        [Fact]
        public void UnsafeFileNameFallsBackToAFixedName()
        {
            // A file name that cannot be an entry (here it has a colon) still gets a safe name.
            Assert.Equal("reference", NativeFiles.EntryNameFor(Root, @"D:\Lib\a:b.sldprt"));
        }

        [Fact]
        public void CollisionsGetUniqueNamesAndAreReported()
        {
            var taken = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            Assert.Equal("Bolt.sldprt", NativeFiles.Unique("Bolt.sldprt", taken, out var first));
            Assert.False(first);

            Assert.Equal("bolt (2).sldprt", NativeFiles.Unique("bolt.sldprt", taken, out var second));
            Assert.True(second);

            Assert.Equal("BOLT (3).sldprt", NativeFiles.Unique("BOLT.sldprt", taken, out var third));
            Assert.True(third);
        }

        [Fact]
        public void CollisionsInsideAFolderKeepTheFolder()
        {
            var taken = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            NativeFiles.Unique("Parts/Bolt.sldprt", taken, out _);

            Assert.Equal("Parts/Bolt (2).sldprt", NativeFiles.Unique("Parts/Bolt.sldprt", taken, out var renamed));
            Assert.True(renamed);
        }

        [Fact]
        public void CollisionsOfNamesWithoutAnExtensionGetASuffix()
        {
            var taken = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "README" };

            Assert.Equal("README (2)", NativeFiles.Unique("README", taken, out var renamed));
            Assert.True(renamed);
        }
    }
}
