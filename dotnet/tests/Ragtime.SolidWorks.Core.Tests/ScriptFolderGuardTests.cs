using System;
using System.IO;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptFolderGuardTests
    {
        private const string AppFolder = @"C:\Users\u\AppData\Local\RagtimeSolidWorks";
        private const string Root = AppFolder + @"\scripts";
        private static readonly FileAttributes? Plain = FileAttributes.Directory;
        private static readonly FileAttributes? Linked = FileAttributes.Directory | FileAttributes.ReparsePoint;

        [Fact]
        public void PlainAncestorAndRootAreNotLinked()
        {
            var linked = ScriptFolderGuard.FirstLinkedFolder(new[] { AppFolder, Root }, _ => Plain);

            Assert.Null(linked);
        }

        [Fact]
        public void LinkedRootIsReported()
        {
            var linked = ScriptFolderGuard.FirstLinkedFolder(new[] { AppFolder, Root }, path => path == Root ? Linked : Plain);

            Assert.Equal(Root, linked);
        }

        [Fact]
        public void LinkedAncestorIsReportedWhenRootIsPlain()
        {
            var linked = ScriptFolderGuard.FirstLinkedFolder(new[] { AppFolder, Root }, path => path == AppFolder ? Linked : Plain);

            Assert.Equal(AppFolder, linked);
        }

        [Fact]
        public void MissingFolderIsNotALink()
        {
            var linked = ScriptFolderGuard.FirstLinkedFolder(new[] { AppFolder, Root }, _ => null);

            Assert.Null(linked);
        }

        [Fact]
        public void FirstLinkInOrderWins()
        {
            var linked = ScriptFolderGuard.FirstLinkedFolder(new[] { AppFolder, Root }, _ => Linked);

            Assert.Equal(AppFolder, linked);
        }

        [Fact]
        public void LookupFailureIsNotSwallowed()
        {
            Assert.Throws<IOException>(() => ScriptFolderGuard.FirstLinkedFolder(new[] { Root }, _ => throw new IOException("denied")));
        }
    }
}
