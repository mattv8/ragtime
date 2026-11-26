using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>A private work folder under the system temp path, removed when the test ends.</summary>
    public sealed class TempWorkFolder : IDisposable
    {
        public TempWorkFolder()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ragtime-sw-test-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, true);
            }
        }
    }

    public sealed class ModelFilesTests : IDisposable
    {
        private const string PartPath = @"C:\Models\Bracket.sldprt";
        private const string AssemblyPath = @"C:\Models\Frame.sldasm";
        private const string DrawingPath = @"C:\Models\Sheet.slddrw";

        private readonly TempWorkFolder _work = new TempWorkFolder();

        public void Dispose() => _work.Dispose();

        [Fact]
        public async Task CleanupWorksWhenWorkFolderHasLinkedAncestor()
        {
            var real = Path.Combine(_work.Path, "real");
            var link = Path.Combine(_work.Path, "link");
            Directory.CreateDirectory(real);
            try
            {
                Directory.CreateSymbolicLink(link, real);
            }
            catch (Exception error) when (error is IOException || error is UnauthorizedAccessException || error is PlatformNotSupportedException)
            {
                // Symlink creation may require privileges unavailable to this test.
                return;
            }

            var h = await OpenedHarness(link);
            string? stray = null;
            h.Manager.SessionFor("i1").OnFileWrite = target =>
            {
                stray = Path.Combine(link, Path.GetFileNameWithoutExtension(target) + " - box.STL");
                File.WriteAllText(stray, "stray");
            };

            var response = await h.CallAsync("model.export", Harness.Args(("format", Harness.S("stl")), ("include_preview", WireJson.Value(false))));

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.NotNull(stray);
            Assert.False(File.Exists(stray));
        }

        [Fact]
        public void FileOperationSchemasDescribeOptionalDocumentAndFormatLimits()
        {
            var export = OperationRegistry.Default.Find("model.export")!.ArgumentsSchema;
            var preview = OperationRegistry.Default.Find("model.preview")!.ArgumentsSchema;
            var summary = OperationRegistry.Default.Find("model.summary")!.ArgumentsSchema;
            var shared = (string)summary["properties"]!["document"]!["description"]!;
            var expected = shared + " Optional when exactly one document is open.";
            Assert.Equal(expected, (string)export["properties"]!["document"]!["description"]!);
            Assert.Equal(expected, (string)preview["properties"]!["document"]!["description"]!);
            var format = (string)export["properties"]!["format"]!["description"]!;
            Assert.Contains("Drawings export as native only", format);
            Assert.Contains("assemblies support step, 3mf, glb and native, not stl", format);
        }

        [Theory]
        [InlineData("model.export", false)]
        [InlineData("model.export", true)]
        [InlineData("model.preview", false)]
        [InlineData("model.preview", true)]
        public async Task StrayFilesAreRemovedAfterEachAttempt(string operation, bool fail)
        {
            var h = await OpenedHarness(_work.Path);
            var session = h.Manager.SessionFor("i1");
            session.FailExport = fail;
            session.FailPreview = fail;
            var unrelated = Path.Combine(_work.Path, "unrelated.stl");
            File.WriteAllText(unrelated, "keep");
            string? stray = null;
            session.OnFileWrite = target =>
            {
                stray = Path.Combine(_work.Path, Path.GetFileNameWithoutExtension(target) + " - box-1.STL");
                File.WriteAllText(stray, "stray");
            };
            var args = operation == "model.export"
                ? Harness.Args(("format", Harness.S("stl")), ("include_preview", WireJson.Value(false)))
                : Harness.Args();

            var response = await h.CallAsync(operation, args);

            Assert.Equal(!fail, (bool)response["ok"]!);
            Assert.NotNull(stray);
            Assert.False(File.Exists(stray));
            Assert.True(File.Exists(unrelated));
            if (!fail)
            {
                Assert.True(File.Exists((string)response["result"]!["files"]![0]!["path"]!));
            }
        }

        [Fact]
        public async Task CleanupDoesNotFollowOrDeleteStraySymlink()
        {
            var h = await OpenedHarness(_work.Path);
            var original = Path.Combine(_work.Path, "unrelated.stl");
            File.WriteAllText(original, "keep");
            string? link = null;
            var symlinkUnavailable = false;
            h.Manager.SessionFor("i1").OnFileWrite = target =>
            {
                link = Path.Combine(_work.Path, Path.GetFileNameWithoutExtension(target) + " - link.STL");
                try
                {
                    File.CreateSymbolicLink(link, original);
                }
                catch (Exception error) when (error is IOException || error is UnauthorizedAccessException || error is PlatformNotSupportedException)
                {
                    symlinkUnavailable = true;
                }
            };

            var response = await h.CallAsync("model.export", Harness.Args(("format", Harness.S("stl")), ("include_preview", WireJson.Value(false))));

            // Symlink creation may be unavailable without platform privileges.
            if (symlinkUnavailable) return;
            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Equal("keep", File.ReadAllText(original));
            Assert.True(File.Exists(link));
        }

        [Theory]
        [InlineData("model.export", "doc-1")]
        [InlineData("model.export", PartPath)]
        [InlineData("model.preview", "doc-1")]
        [InlineData("model.preview", PartPath)]
        public async Task FileOperationResolvesDocumentIdOrPath(string operation, string document)
        {
            var h = await OpenedHarness(_work.Path);
            var args = operation == "model.export"
                ? Harness.Args(("document", Harness.S(document)), ("format", Harness.S("step")), ("include_preview", WireJson.Value(false)))
                : Harness.Args(("document", Harness.S(document)));

            var response = await h.CallAsync(operation, args);

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Equal(PartPath, (string?)response["result"]!["document"]!["path"]);
        }

        [Theory]
        [InlineData("model.export", "slddrw")]
        [InlineData("model.preview", "png")]
        public async Task DrawingFileNameUsesPathInsteadOfSheetTitle(string operation, string extension)
        {
            var h = new Harness(workFolder: _work.Path);
            h.Manager.AddBackground("i1");
            h.Manager.SessionFor("i1").Open.Add(new SessionDocumentInfo("box - Sheet1", @"C:\Models\box.slddrw", "drawing", null, false, false));
            var args = operation == "model.export"
                ? Harness.Args(("format", Harness.S("native")))
                : Harness.Args();

            var response = await h.CallAsync(operation, args);

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Equal("box." + extension, (string)response["result"]!["files"]![0]!["file_name"]!);
        }

        [Fact]
        public async Task UnsavedPreviewFileNameFallsBackToTitle()
        {
            var h = new Harness(workFolder: _work.Path);
            h.Manager.AddBackground("i1");
            h.Manager.SessionFor("i1").Open.Add(new SessionDocumentInfo("Drawing1 - Sheet1", "", "drawing", null, false, true));

            var response = await h.CallAsync("model.preview", Harness.Args());

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Equal("Drawing1 - Sheet1.png", (string)response["result"]!["files"]![0]!["file_name"]!);
        }

        private static JObject Export(string path, string format, bool? includePreview = null, string? configuration = null)
        {
            var pairs = new System.Collections.Generic.List<(string, JToken)>
            {
                ("document", Harness.S(path)),
                ("format", Harness.S(format)),
            };
            if (includePreview.HasValue)
            {
                pairs.Add(("include_preview", WireJson.Value(includePreview.Value)));
            }

            if (configuration != null)
            {
                pairs.Add(("configuration", Harness.S(configuration)));
            }

            return Harness.Args(pairs.ToArray());
        }

        private static JObject Preview(params (string Name, JToken Value)[] pairs) => Harness.Args(pairs);

        private static async Task<Harness> OpenedHarness(string workFolder, string path = PartPath)
        {
            var h = new Harness(workFolder: workFolder);
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(path))));
            return h;
        }

        private static string Sha256Hex(string path)
        {
            using (var sha = SHA256.Create())
            {
                return BitConverter.ToString(sha.ComputeHash(File.ReadAllBytes(path))).Replace("-", string.Empty).ToLowerInvariant();
            }
        }

        [Fact]
        public void CatalogListsBothOperationsAsReadAfterEquations()
        {
            var names = OperationRegistry.Default.Operations.Select(o => o.Name).ToList();

            var export = OperationRegistry.Default.Find("model.export")!;
            var preview = OperationRegistry.Default.Find("model.preview")!;
            Assert.Equal(ProtocolInfo.Access.Read, export.Access);
            Assert.Equal(ProtocolInfo.Access.Read, preview.Access);
            Assert.True(names.IndexOf("model.export") > names.IndexOf("equations.list"));
            Assert.True(names.IndexOf("model.preview") > names.IndexOf("model.export"));
            Assert.Equal(33, OperationRegistry.Default.Public.Count);
        }

        [Fact]
        public async Task ExportWritesStepInsideWorkFolderWithNameSizeAndHash()
        {
            var h = await OpenedHarness(_work.Path);

            var response = await h.CallAsync("model.export", Export(PartPath, "step", includePreview: false));

            Assert.True((bool)response["ok"]!, response.ToString());
            var file = response["result"]!["files"]![0]!;
            var path = (string)file["path"]!;
            Assert.Equal("model", (string)file["role"]!);
            Assert.Equal("step", (string)file["format"]!);
            Assert.Equal("Bracket.step", (string)file["file_name"]!);
            Assert.StartsWith(_work.Path, path, StringComparison.OrdinalIgnoreCase);
            Assert.Matches(@"^export-[0-9a-f]{32}\.step$", Path.GetFileName(path));
            Assert.Equal(new FileInfo(path).Length, (long)file["size_bytes"]!);
            Assert.Equal(Sha256Hex(path), (string)file["sha256"]!);
            Assert.Equal(path, h.Manager.SessionFor("i1").Exports.Single().TargetPath);
        }

        [Fact]
        public async Task DocumentFieldsAndConfigurationAreReported()
        {
            var h = await OpenedHarness(_work.Path);

            var response = await h.CallAsync("model.export", Export(PartPath, "stl", includePreview: false, configuration: "Small"));

            var document = response["result"]!["document"]!;
            Assert.Equal(PartPath, (string)document["path"]!);
            Assert.Equal("part", (string)document["type"]!);
            Assert.Equal("Small", (string)document["configuration"]!);
            Assert.Equal("Small", h.Manager.SessionFor("i1").Exports.Single().Configuration);
        }

        [Fact]
        public async Task IncludePreviewIsOnByDefaultAndAddsPreviewEntry()
        {
            var h = await OpenedHarness(_work.Path);

            var response = await h.CallAsync("model.export", Export(PartPath, "3mf"));

            var files = (JArray)response["result"]!["files"]!;
            Assert.Equal(2, files.Count);
            Assert.Equal("preview", (string)files[1]!["role"]!);
            Assert.Equal("png", (string)files[1]!["format"]!);
            Assert.Equal("Bracket.png", (string)files[1]!["file_name"]!);
            Assert.Equal(Sha256Hex((string)files[1]!["path"]!), (string)files[1]!["sha256"]!);
            var preview = h.Manager.SessionFor("i1").Previews.Single();
            Assert.Equal("isometric", preview.View);
            Assert.Equal(1024, preview.Width);
            Assert.Equal(768, preview.Height);
        }

        [Fact]
        public async Task IncludePreviewFalseSkipsThePreview()
        {
            var h = await OpenedHarness(_work.Path);

            await h.CallAsync("model.export", Export(PartPath, "glb", includePreview: false));

            Assert.Empty(h.Manager.SessionFor("i1").Previews);
        }

        [Fact]
        public async Task PreviewFailureIsWarningAndExportStillSucceeds()
        {
            var h = await OpenedHarness(_work.Path);
            h.Manager.SessionFor("i1").FailPreview = true;

            var response = await h.CallAsync("model.export", Export(PartPath, "step"));

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Single((JArray)response["result"]!["files"]!);
            Assert.Contains((JArray)response["result"]!["warnings"]!, w => ((string?)w)!.StartsWith("No preview was made", StringComparison.Ordinal));
        }

        [Fact]
        public async Task SessionWarningsAreForwarded()
        {
            var h = await OpenedHarness(_work.Path);
            h.Manager.SessionFor("i1").ExportWarnings.Add("The document has unsaved changes.");

            var response = await h.CallAsync("model.export", Export(PartPath, "step", includePreview: false));

            Assert.Equal("The document has unsaved changes.", (string?)response["result"]!["warnings"]![0]);
        }

        [Fact]
        public async Task UnknownFormatIsInvalidArguments()
        {
            var h = await OpenedHarness(_work.Path);

            var response = await h.CallAsync("model.export", Export(PartPath, "iges"));

            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
            Assert.Empty(h.Manager.SessionFor("i1").Exports);
        }

        [Fact]
        public async Task DrawingRefusesModelFormatsButExportsNative()
        {
            var h = await OpenedHarness(_work.Path, DrawingPath);

            var refused = await h.CallAsync("model.export", Export(DrawingPath, "step"));
            var native = await h.CallAsync("model.export", Export(DrawingPath, "native", includePreview: false));

            Assert.Equal("invalid_arguments", (string?)refused["error"]!["code"]);
            Assert.True((bool)native["ok"]!, native.ToString());
            Assert.EndsWith(".slddrw", (string)native["result"]!["files"]![0]!["path"]!);
            Assert.Single(h.Manager.SessionFor("i1").Exports);
        }

        [Fact]
        public async Task AssemblyStlIsRejectedBeforeCallingSession()
        {
            var h = await OpenedHarness(_work.Path, AssemblyPath);

            var response = await h.CallAsync("model.export", Export(AssemblyPath, "stl", includePreview: false));

            Assert.False((bool)response["ok"]!, response.ToString());
            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
            Assert.Equal("format", (string?)response["error"]!["details"]!["field"]);
            Assert.Equal("STL export of an assembly writes one file per component. Export the assembly as 3mf or glb for a single mesh, or step.", (string?)response["error"]!["message"]);
            Assert.Empty(h.Manager.SessionFor("i1").Exports);
        }

        [Fact]
        public async Task PartStlRemainsAvailable()
        {
            var h = await OpenedHarness(_work.Path);

            var response = await h.CallAsync("model.export", Export(PartPath, "stl", includePreview: false));

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Single(h.Manager.SessionFor("i1").Exports);
        }

        [Fact]
        public async Task NativeAssemblyIsZipNamedAfterTheDocument()
        {
            var h = await OpenedHarness(_work.Path, AssemblyPath);

            var response = await h.CallAsync("model.export", Export(AssemblyPath, "native", includePreview: false));

            var file = response["result"]!["files"]![0]!;
            Assert.Equal("native", (string)file["role"]!);
            Assert.Equal("zip", (string)file["format"]!);
            Assert.Equal("Frame.zip", (string)file["file_name"]!);
            Assert.Equal("assembly", (string)response["result"]!["document"]!["type"]!);
        }

        [Fact]
        public async Task DisplayNameIsSanitisedFromTheTitle()
        {
            var odd = @"C:\Models\Bad (copy) #2 & co.sldprt";
            var h = await OpenedHarness(_work.Path, odd);

            var response = await h.CallAsync("model.export", Export(odd, "step", includePreview: false));

            Assert.Equal("Bad (copy) _2 _ co.step", (string)response["result"]!["files"]![0]!["file_name"]!);
        }

        [Fact]
        public async Task FailedExportRemovesThePartialFile()
        {
            var h = await OpenedHarness(_work.Path);
            h.Manager.SessionFor("i1").FailExport = true;

            var response = await h.CallAsync("model.export", Export(PartPath, "step"));

            Assert.Equal("export_failed", (string?)response["error"]!["code"]);
            Assert.Empty(Directory.GetFiles(_work.Path));
        }

        [Fact]
        public async Task MissingWorkFolderIsInternalErrorBeforeAnyCall()
        {
            var h = new Harness(workFolder: null);
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(PartPath))));

            var response = await h.CallAsync("model.export", Export(PartPath, "step"));

            Assert.Equal("internal_error", (string?)response["error"]!["code"]);
            Assert.Empty(h.Manager.SessionFor("i1").Exports);
        }

        [Fact]
        public async Task PreviewSizeIsClampedToTheAllowedRange()
        {
            var h = await OpenedHarness(_work.Path);

            await h.CallAsync("model.preview", Preview(("document", Harness.S(PartPath)), ("width", Harness.I(5000)), ("height", Harness.I(10))));

            var preview = h.Manager.SessionFor("i1").Previews.Single();
            Assert.Equal(2048, preview.Width);
            Assert.Equal(256, preview.Height);
        }

        [Fact]
        public async Task PreviewReturnsOnePngWithFacts()
        {
            var h = await OpenedHarness(_work.Path);

            var response = await h.CallAsync("model.preview", Preview(("document", Harness.S(PartPath)), ("view", Harness.S("front"))));

            Assert.True((bool)response["ok"]!, response.ToString());
            var files = (JArray)response["result"]!["files"]!;
            Assert.Single(files);
            Assert.Equal("preview", (string)files[0]!["role"]!);
            Assert.Equal("png", (string)files[0]!["format"]!);
            Assert.Equal("front", h.Manager.SessionFor("i1").Previews.Single().View);
        }

        [Fact]
        public async Task PreviewFailureIsPreviewUnavailable()
        {
            var h = await OpenedHarness(_work.Path);
            h.Manager.SessionFor("i1").FailPreview = true;

            var response = await h.CallAsync("model.preview", Preview(("document", Harness.S(PartPath))));

            Assert.Equal("preview_unavailable", (string?)response["error"]!["code"]);
            Assert.Empty(Directory.GetFiles(_work.Path));
        }

        [Fact]
        public async Task ExportPreviewUsesTheExportedConfiguration()
        {
            var h = await OpenedHarness(_work.Path);

            await h.CallAsync("model.export", Export(PartPath, "step", includePreview: true, configuration: "Small"));

            Assert.Equal("Small", h.Manager.SessionFor("i1").Previews.Single().Configuration);
        }

        [Fact]
        public async Task PreviewOperationShowsTheActiveConfiguration()
        {
            var h = await OpenedHarness(_work.Path);

            await h.CallAsync("model.preview", Preview(("document", Harness.S(PartPath))));

            Assert.Null(h.Manager.SessionFor("i1").Previews.Single().Configuration);
        }

        [Fact]
        public async Task NativeExportWithConfigurationWarnsThatItIsIgnored()
        {
            var h = await OpenedHarness(_work.Path);

            var response = await h.CallAsync("model.export", Export(PartPath, "native", includePreview: false, configuration: "Small"));

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Contains((JArray)response["result"]!["warnings"]!, w => ((string?)w)!.StartsWith("The configuration is not used for native files", StringComparison.Ordinal));
        }

        [Fact]
        public async Task UnexpectedFaultRemovesEveryFileOfTheRequest()
        {
            var h = await OpenedHarness(_work.Path);
            h.Manager.SessionFor("i1").PreviewException = new InvalidOperationException("boom C:\\secret");

            var response = await h.CallAsync("model.export", Export(PartPath, "step"));

            Assert.Equal("internal_error", (string?)response["error"]!["code"]);
            Assert.DoesNotContain("secret", response.ToString());
            Assert.Empty(Directory.GetFiles(_work.Path));
        }

        [Theory]
        [InlineData("model.export")]
        [InlineData("model.preview")]
        public async Task DocumentIsOptionalWhenExactlyOneDocumentIsOpen(string operation)
        {
            var h = await OpenedHarness(_work.Path);

            var args = operation == "model.export"
                ? Harness.Args(("format", Harness.S("step")), ("include_preview", WireJson.Value(false)))
                : Harness.Args();
            var response = await h.CallAsync(operation, args);

            Assert.True((bool)response["ok"]!, response.ToString());
            Assert.Equal(PartPath, (string?)response["result"]!["document"]!["path"]);
        }

        [Theory]
        [InlineData("model.export")]
        [InlineData("model.preview")]
        public async Task DocumentIsRequiredWhenSeveralDocumentsAreOpen(string operation)
        {
            var h = await OpenedHarness(_work.Path);
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(AssemblyPath))));

            var args = operation == "model.export" ? Harness.Args(("format", Harness.S("step"))) : Harness.Args();
            var response = await h.CallAsync(operation, args);

            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
            Assert.Equal("document", (string?)response["error"]!["details"]!["field"]);
        }

        [Theory]
        [InlineData("model.export")]
        [InlineData("model.preview")]
        public async Task FileOperationRejectsOldPathArgument(string operation)
        {
            var h = await OpenedHarness(_work.Path);
            var args = operation == "model.export"
                ? Harness.Args(("path", Harness.S(PartPath)), ("format", Harness.S("step")))
                : Harness.Args(("path", Harness.S(PartPath)));

            var response = await h.CallAsync(operation, args);

            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
            Assert.Equal("path", (string?)response["error"]!["details"]!["field"]);
        }
    }
}
