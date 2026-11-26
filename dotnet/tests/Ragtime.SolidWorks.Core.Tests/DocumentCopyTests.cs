using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>document.copy through the processor: folder, file list, limits, cleanup and routing (contract §5.5).</summary>
    public class DocumentCopyTests : IDisposable
    {
        private const string AssemblyPath = @"C:\Models\Asm.sldasm";

        private readonly string _work = Path.Combine(Path.GetTempPath(), "ragtime-sw-copy-" + Guid.NewGuid().ToString("N"));

        public DocumentCopyTests() => Directory.CreateDirectory(_work);

        public void Dispose()
        {
            if (Directory.Exists(_work))
            {
                Directory.Delete(_work, recursive: true);
            }
        }

        private string CopiesRoot => Path.Combine(_work, "copies");

        /// <summary>A harness with the assembly open as doc-1 in background instance i1.</summary>
        private async Task<Harness> OpenedHarness(string? workFolder = null, System.Action<FakeSession>? seed = null)
        {
            var h = new Harness(workFolder: workFolder ?? _work);
            h.Manager.AddBackground("i1");
            seed?.Invoke(h.Manager.SessionFor("i1"));
            Assert.True((bool)(await h.CallAsync("document.open", Harness.Args(("path", Harness.S(AssemblyPath)))))["ok"]!);
            return h;
        }

        private static JObject Copy(Harness h, JObject? arguments = null) =>
            h.CallAsync("document.copy", arguments ?? Harness.Args(("document", Harness.S("doc-1")))).GetAwaiter().GetResult();

        [Fact]
        public async Task CopyReturnsTheFolderTopPathAndEveryFileInsideTheCopiesTree()
        {
            var h = await OpenedHarness();

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.True((bool)response["ok"]!, response.ToString());
            var result = response["result"]!;
            var folder = (string?)result["folder"]!;
            Assert.StartsWith(CopiesRoot + Path.DirectorySeparatorChar, folder);
            Assert.True(Directory.Exists(folder));
            Assert.StartsWith(folder + Path.DirectorySeparatorChar, (string?)result["path"]!);
            Assert.Equal(3, (int)result["count"]!);
            var files = (JArray)result["files"]!;
            Assert.Equal(3, files.Count);
            Assert.Equal(AssemblyPath, (string?)files[0]["source"]);
            Assert.Equal((string?)result["path"], (string?)files[0]["path"]);
            Assert.All(files, file => Assert.True(File.Exists((string?)file["path"])));
            Assert.Empty((JArray)result["warnings"]!);
        }

        [Fact]
        public async Task IncludeReferencesFalseCopiesOnlyTheTopDocument()
        {
            var h = await OpenedHarness();

            var result = Copy(h, Harness.Args(("document", Harness.S("doc-1")), ("include_references", Harness.B(false))))["result"]!;

            Assert.Equal(1, (int)result["count"]!);
            Assert.False(h.Manager.SessionFor("i1").CopyRequests.Single().IncludeReferences);
        }

        [Fact]
        public async Task TheSessionReceivesAGeneratedFolderAndTheFiveHundredLimit()
        {
            var h = await OpenedHarness();

            Copy(h);

            var request = h.Manager.SessionFor("i1").CopyRequests.Single();
            Assert.StartsWith(CopiesRoot + Path.DirectorySeparatorChar, request.FolderPath);
            Assert.Equal(500, request.MaxFiles);
            Assert.True(request.IncludeReferences);
            Assert.Equal(AssemblyPath, request.Document.Path);
        }

        [Theory]
        [InlineData(499)]
        [InlineData(500)]
        public async Task LargeCopiesWithinTheLimitSucceedAndListFiftyFiles(int files)
        {
            var h = await OpenedHarness(seed: session => session.CopyFileCount = files);

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.True((bool)response["ok"]!, response.ToString());
            var result = response["result"]!;
            Assert.Equal(files, (int)result["count"]!);
            Assert.Equal(50, ((JArray)result["files"]!).Count);
            Assert.True((bool)result["files_truncated"]!);
            Assert.Equal(AssemblyPath, (string?)((JArray)result["files"]!)[0]["source"]);
        }

        [Theory]
        [InlineData(3, false, 3)]
        [InlineData(50, false, 50)]
        [InlineData(51, true, 50)]
        public async Task FilesTruncatedIsFalseUntilMoreThanFiftyFilesExist(int files, bool truncated, int listed)
        {
            var h = await OpenedHarness(seed: session => session.CopyFileCount = files);

            var result = Copy(h)["result"]!;

            Assert.Equal(files, (int)result["count"]!);
            Assert.Equal(truncated, (bool)result["files_truncated"]!);
            Assert.Equal(listed, ((JArray)result["files"]!).Count);
        }

        [Fact]
        public async Task MoreThanFiveHundredFilesIsRefusedWithTheFileCountDetails()
        {
            var h = await OpenedHarness(seed: session => session.CopyFileCount = 501);

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.False((bool)response["ok"]!);
            Assert.Equal("result_too_large", (string?)response["error"]!["code"]);
            var details = response["error"]!["details"]!;
            Assert.Equal("file_count", (string?)details["reason"]);
            Assert.Equal(500, (int)details["limit"]!);
            Assert.Equal(501, (int)details["count"]!);
        }

        [Fact]
        public async Task CoreRefusesAFileListLongerThanTheLimitAndRemovesTheCopy()
        {
            var h = await OpenedHarness(seed: session =>
            {
                session.CopyFileCount = 501;
                session.IgnoreCopyFileLimit = true;
            });

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.Equal("result_too_large", (string?)response["error"]!["code"]);
            Assert.Equal("file_count", (string?)response["error"]!["details"]!["reason"]);
            Assert.False(Directory.Exists(CopiesRoot) && Directory.EnumerateDirectories(CopiesRoot).Any());
        }

        [Fact]
        public async Task SessionFailureRemovesTheCopyFolder()
        {
            var h = await OpenedHarness(seed: session => session.FailCopy = true);

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.Equal("solidworks_error", (string?)response["error"]!["code"]);
            Assert.False(Directory.Exists(CopiesRoot) && Directory.EnumerateDirectories(CopiesRoot).Any());
        }

        [Fact]
        public async Task WarningsFromTheSessionArePassedThrough()
        {
            var h = await OpenedHarness(seed: session => session.CopyWarnings.Add("Missing source: C:\\Parts\\Gone.sldprt"));

            var result = Copy(h)["result"]!;

            Assert.Equal(new[] { "Missing source: C:\\Parts\\Gone.sldprt" }, ((JArray)result["warnings"]!).Select(warning => (string?)warning));
        }

        [Fact]
        public async Task CopyNeedsAnOpenDocument()
        {
            var h = await OpenedHarness();

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S(@"C:\Models\Other.sldasm"))));

            Assert.Equal("document_not_open", (string?)response["error"]!["code"]);
            Assert.Empty(h.Manager.SessionFor("i1").CopyRequests);
        }

        [Fact]
        public async Task CopyWithoutAWorkFolderIsRefusedBeforeAnyInstanceCall()
        {
            var h = new Harness(workFolder: null);
            h.Manager.AddBackground("i1");
            Assert.True((bool)(await h.CallAsync("document.open", Harness.Args(("path", Harness.S(AssemblyPath)))))["ok"]!);

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S("doc-1"))));

            Assert.Equal("internal_error", (string?)response["error"]!["code"]);
            Assert.Empty(h.Manager.SessionFor("i1").CopyRequests);
        }

        [Fact]
        public async Task CopyRejectsUnknownArguments()
        {
            var h = await OpenedHarness();

            var response = await h.CallAsync("document.copy", Harness.Args(("document", Harness.S("doc-1")), ("folder", Harness.S(@"C:\x"))));

            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
            Assert.Equal("folder", (string?)response["error"]!["details"]!["field"]);
        }
    }
}
