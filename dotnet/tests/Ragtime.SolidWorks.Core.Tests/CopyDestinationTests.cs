using System;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>
    /// Destination confinement and non-local source refusal of document.copy (review findings 1 and the source rule). These
    /// are the pure rules the Readers lane calls before any save; the reparse-point and Win32 checks run in Readers.
    /// </summary>
    public class CopyDestinationTests
    {
        private const string Folder = @"C:\Work\copies\0123456789abcdef0123456789abcdef";

        [Theory]
        [InlineData(@"C:\a\b.sldasm", @"C:\a\b.sldasm")]
        [InlineData(@"c:\a\b.sldasm", @"C:\a\b.sldasm")]
        [InlineData(@"C:\a\.\b.sldasm", @"C:\a\b.sldasm")]
        [InlineData(@"C:\a\b\..\c.sldasm", @"C:\a\c.sldasm")]
        [InlineData(@"C:\a\\b.sldasm", @"C:\a\b.sldasm")]
        public void CanonicalFormResolvesDotSegmentsAndDriveCase(string path, string expected)
        {
            Assert.Equal(expected, CopyDestination.Canonical(path));
        }

        [Theory]
        [InlineData(@"C:\..\x.sldasm")]
        [InlineData(@"C:\a\..\..\x.sldasm")]
        [InlineData(@"relative\x.sldasm")]
        [InlineData(@"\\server\share\x.sldasm")]
        [InlineData(@"C:x.sldasm")]
        [InlineData(@"C:\a/b.sldasm")]
        [InlineData(@"C:\a:b.sldasm")]
        [InlineData(null)]
        [InlineData("")]
        public void AnyPathThatIsNotAbsoluteOnADriveOrEscapesTheRootIsRefused(string? path)
        {
            AssertInvalid(() => CopyDestination.Canonical(path));
        }

        [Fact]
        public void TargetsInsideTheFolderAreAccepted()
        {
            CopyDestination.RequireInside(Folder, new[] { Folder + @"\Asm.sldasm", Folder + @"\sub\Part.sldprt" });
        }

        [Fact]
        public void ADotDotThatEscapesTheFolderIsRefusedBeforeAnySave()
        {
            AssertInvalid(() => CopyDestination.RequireInside(Folder, new[] { Folder + @"\..\outside\Asm.sldasm" }));
        }

        [Fact]
        public void ADotDotThatStaysInsideTheFolderIsAccepted()
        {
            CopyDestination.RequireInside(Folder, new[] { Folder + @"\sub\..\Asm.sldasm" });
        }

        [Fact]
        public void APrefixSiblingFolderIsNotInside()
        {
            // "...\0123...def2" starts with the folder's text but is a different folder.
            AssertInvalid(() => CopyDestination.RequireInside(Folder, new[] { Folder + "2\\Asm.sldasm" }));
        }

        [Fact]
        public void TheFolderItselfIsNotAnInsideTarget()
        {
            AssertInvalid(() => CopyDestination.RequireInside(Folder, new[] { Folder }));
        }

        [Fact]
        public void OneOutsideTargetRefusesTheWholeCopy()
        {
            AssertInvalid(() => CopyDestination.RequireInside(Folder, new[] { Folder + @"\Asm.sldasm", @"C:\Models\Other.sldprt" }));
        }

        [Fact]
        public void ANullTargetIsRefused()
        {
            AssertInvalid(() => CopyDestination.RequireInside(Folder, new string?[] { null }));
        }

        [Fact]
        public void TheCheckIgnoresDriveLetterAndFolderCase()
        {
            CopyDestination.RequireInside(Folder, new[] { @"c:\work\COPIES\0123456789ABCDEF0123456789ABCDEF\Asm.sldasm" });
        }

        [Theory]
        [InlineData(@"\\server\share\Asm.sldasm")]
        [InlineData(@"\\?\C:\Models\Asm.sldasm")]
        [InlineData(@"relative\Asm.sldasm")]
        [InlineData(@"C:\Models\a|b.sldasm")]
        public void ANonLocalSourceIsRefusedBeforeAnySave(string source)
        {
            // The same path rules as document.open. A UNC or device source (or one with a bad character) stops the copy here.
            var error = Assert.Throws<HostErrorException>(() => CopyDestination.RequireLocalSources(new[] { source }));
            Assert.Equal(ErrorCodes.InvalidArguments, error.Code);
        }

        [Fact]
        public void LocalSourcesPassTheSourceRule()
        {
            CopyDestination.RequireLocalSources(new[] { @"C:\Models\Asm.sldasm", @"D:\Parts\Part.sldprt" });
        }

        private static void AssertInvalid(Action action)
        {
            var error = Assert.Throws<HostErrorException>(action);
            Assert.Equal(ErrorCodes.InvalidArguments, error.Code);
        }
    }
}
