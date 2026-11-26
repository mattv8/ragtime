using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>An in-memory file system for the retention sweep. Failures are configured per path.</summary>
    internal sealed class FakeCopyFileSystem : ICopyFileSystem
    {
        public const string Root = @"C:\Work\copies";

        public List<CopyFolder> Folders { get; } = new List<CopyFolder>();

        public HashSet<string> Files { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        /// <summary>Paths whose delete throws (for example a file in use).</summary>
        public HashSet<string> LockedFiles { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        /// <summary>When set, listing the root throws.</summary>
        public bool FailListFolders { get; set; }

        /// <summary>When set, listing a folder's files throws.</summary>
        public bool FailListFiles { get; set; }

        public IReadOnlyList<CopyFolder> ListFolders(string root)
        {
            if (FailListFolders)
            {
                throw new System.IO.IOException("Access denied.");
            }

            return Folders.Where(folder => folder.Path.StartsWith(root + "\\", StringComparison.OrdinalIgnoreCase)
                && folder.Path.IndexOf('\\', root.Length + 1) < 0).ToList();
        }

        public IReadOnlyList<string> ListFiles(string folder)
        {
            if (FailListFiles)
            {
                throw new System.IO.IOException("Access denied.");
            }

            return Files.Where(file => file.StartsWith(folder + "\\", StringComparison.OrdinalIgnoreCase)).ToList();
        }

        public void DeleteFile(string path)
        {
            if (LockedFiles.Contains(path))
            {
                throw new System.IO.IOException("The file is in use.");
            }

            Files.Remove(path);
        }

        public void DeleteFolder(string folder)
        {
            if (Files.Any(file => file.StartsWith(folder + "\\", StringComparison.OrdinalIgnoreCase)))
            {
                throw new System.IO.IOException("The folder is not empty.");
            }

            Folders.RemoveAll(entry => string.Equals(entry.Path, folder, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>A generated copy id for a short test name: the name's ASCII hex, left-padded with zeros to 32 digits.</summary>
        public static string IdOf(string name)
        {
            var hex = string.Concat(System.Text.Encoding.ASCII.GetBytes(name).Select(b => b.ToString("x2")));
            return hex.PadLeft(32, '0');
        }

        /// <summary>The copy folder of a test name: <c>Root\&lt;generated id&gt;</c>.</summary>
        public static string FolderOf(string name) => Root + "\\" + IdOf(name);

        public void AddCopy(string name, DateTime createdUtc, params string[] files)
        {
            var folder = FolderOf(name);
            Folders.Add(new CopyFolder(folder, createdUtc));
            foreach (var file in files)
            {
                Files.Add(folder + "\\" + file);
            }
        }
    }

    /// <summary>The copy retention sweep (contract §5.5): age, open documents, per-file tolerance, listing failures.</summary>
    public class CopyRetentionTests
    {
        private static readonly DateTime Now = new DateTime(2026, 11, 25, 12, 0, 0, DateTimeKind.Utc);

        [Fact]
        public void TheRetentionAgeIsSevenDays()
        {
            Assert.Equal(TimeSpan.FromDays(7), CopyRetention.MaxAge);
        }

        [Fact]
        public void ACopyFolderOfExactlySevenDaysIsKept()
        {
            Assert.False(CopyRetention.IsExpired(Now - TimeSpan.FromDays(7), Now));
        }

        [Fact]
        public void ACopyFolderOlderThanSevenDaysIsExpired()
        {
            Assert.True(CopyRetention.IsExpired(Now - TimeSpan.FromDays(7) - TimeSpan.FromSeconds(1), Now));
        }

        [Fact]
        public void TheCopiesRootIsTheCopiesFolderUnderTheWorkFolder()
        {
            var root = CopyRetention.CopiesRoot(@"C:\Work");

            Assert.EndsWith(CopyRetention.CopiesFolderName, root);
            Assert.Equal("copies", CopyRetention.CopiesFolderName);
        }

        [Fact]
        public void AnExpiredCopyFolderIsDeletedWithItsFiles()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("old", Now.AddDays(-8), "Asm.sldasm", "Part.sldprt");

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(1, result.DeletedFolders);
            Assert.Empty(fs.Folders);
            Assert.Empty(fs.Files);
            Assert.Empty(result.FailedPaths);
        }

        [Fact]
        public void AFreshCopyFolderIsKept()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("fresh", Now.AddDays(-6), "Asm.sldasm");

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(0, result.DeletedFolders);
            Assert.Single(fs.Folders);
            Assert.Single(fs.Files);
        }

        [Fact]
        public void AnExpiredFolderWithAnOpenDocumentIsSkipped()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("open", Now.AddDays(-9), "Asm.sldasm");
            var openPath = FakeCopyFileSystem.FolderOf("open") + @"\Asm.sldasm";

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, new[] { openPath });

            Assert.Equal(0, result.DeletedFolders);
            Assert.Equal(1, result.SkippedOpenFolders);
            Assert.Single(fs.Folders);
            Assert.Single(fs.Files);
        }

        [Fact]
        public void AnOpenDocumentOutsideTheFolderDoesNotProtectIt()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("old", Now.AddDays(-9), "Asm.sldasm");

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, new[] { FakeCopyFileSystem.Root + @"\other\Asm.sldasm" });

            Assert.Equal(1, result.DeletedFolders);
            Assert.Equal(0, result.SkippedOpenFolders);
        }

        [Fact]
        public void AFileThatCannotBeDeletedKeepsItsFolderAndIsRetriedNextTime()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("old", Now.AddDays(-9), "Asm.sldasm", "Locked.sldprt");
            var locked = FakeCopyFileSystem.FolderOf("old") + @"\Locked.sldprt";
            fs.LockedFiles.Add(locked);
            var sweeper = new CopyRetention(fs);

            var first = sweeper.Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(0, first.DeletedFolders);
            Assert.Equal(new[] { locked }, first.FailedPaths);
            Assert.Single(fs.Folders);
            Assert.Single(fs.Files);

            fs.LockedFiles.Clear();
            var second = sweeper.Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(1, second.DeletedFolders);
            Assert.Empty(fs.Folders);
        }

        [Fact]
        public void AFailedFileDoesNotStopTheOtherFoldersBeingSwept()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("a", Now.AddDays(-9), "Locked.sldprt");
            fs.AddCopy("b", Now.AddDays(-9), "Free.sldprt");
            fs.LockedFiles.Add(FakeCopyFileSystem.FolderOf("a") + @"\Locked.sldprt");

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(1, result.DeletedFolders);
            Assert.Single(fs.Folders);
        }

        [Fact]
        public void AFailedListingOfTheRootIsReportedNotThrown()
        {
            var fs = new FakeCopyFileSystem { FailListFolders = true };

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(0, result.DeletedFolders);
            Assert.Equal(new[] { FakeCopyFileSystem.Root }, result.FailedPaths);
        }

        [Fact]
        public void AFailedListingOfAFolderKeepsIt()
        {
            var fs = new FakeCopyFileSystem { FailListFiles = true };
            fs.AddCopy("old", Now.AddDays(-9), "Asm.sldasm");

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(0, result.DeletedFolders);
            Assert.Equal(new[] { FakeCopyFileSystem.FolderOf("old") }, result.FailedPaths);
            Assert.Single(fs.Folders);
        }

        [Fact]
        public void TheSweepRequiresTheOpenDocumentList()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new CopyRetention(new FakeCopyFileSystem()).Sweep(FakeCopyFileSystem.Root, Now, null!));
        }

        [Fact]
        public void AnUnreadableOpenDocumentPathMakesTheSweepDeleteNothing()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("old", Now.AddDays(-9), "Asm.sldasm");
            fs.AddCopy("older", Now.AddDays(-30), "Other.sldasm");

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, new string?[] { null });

            Assert.False(result.OpenStateKnown);
            Assert.Equal(0, result.DeletedFolders);
            Assert.Equal(2, result.SkippedOpenFolders);
            Assert.Equal(2, fs.Folders.Count);
            Assert.Equal(2, fs.Files.Count);
            Assert.Empty(result.FailedPaths);
        }

        [Fact]
        public void AKnownOpenStateReportsItsFlag()
        {
            var fs = new FakeCopyFileSystem();
            fs.AddCopy("old", Now.AddDays(-9), "Asm.sldasm");

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, new string?[] { string.Empty });

            Assert.True(result.OpenStateKnown);
            Assert.Equal(1, result.DeletedFolders);
        }

        [Fact]
        public void AFolderWhoseNameIsNotAGeneratedIdIsNeverDeleted()
        {
            var fs = new FakeCopyFileSystem();
            fs.Folders.Add(new CopyFolder(FakeCopyFileSystem.Root + @"\keep-me", Now.AddDays(-60)));
            fs.Files.Add(FakeCopyFileSystem.Root + @"\keep-me\Asm.sldasm");
            fs.Folders.Add(new CopyFolder(FakeCopyFileSystem.Root + @"\" + new string('g', 32), Now.AddDays(-60)));
            fs.Folders.Add(new CopyFolder(FakeCopyFileSystem.Root + @"\" + new string('a', 31), Now.AddDays(-60)));

            var result = new CopyRetention(fs).Sweep(FakeCopyFileSystem.Root, Now, Array.Empty<string>());

            Assert.Equal(0, result.DeletedFolders);
            Assert.Equal(3, fs.Folders.Count);
            Assert.Empty(result.FailedPaths);
        }

        [Theory]
        [InlineData(@"C:\Work\copies\0123456789abcdef0123456789ABCDEF", true)]
        [InlineData(@"C:\Work\copies\0123456789abcdef0123456789abcde", false)]
        [InlineData(@"C:\Work\copies\0123456789abcdef0123456789abcdef0", false)]
        [InlineData(@"C:\Work\copies\0123456789abcdef0123456789abcdeg", false)]
        [InlineData(@"C:\Work\copies\keep-me", false)]
        [InlineData(@"C:\Work\copies\..", false)]
        public void OnlyFoldersNamedWithAGeneratedIdAreCopyFolders(string folder, bool expected)
        {
            Assert.Equal(expected, CopyRetention.IsGeneratedFolder(folder));
        }

        [Fact]
        public void AGeneratedIdFolderWithATrailingSeparatorIsRecognised()
        {
            Assert.True(CopyRetention.IsGeneratedFolder(@"C:\Work\copies\0123456789abcdef0123456789abcdef\"));
        }
    }
}
