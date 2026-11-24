using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>The pdm.* operations through <see cref="RequestProcessor.ProcessLineAsync"/> with the fake gateway.</summary>
    public class PdmOperationTests
    {
        private const string Approved = @"\13_Ragtime";
        private const string ApprovedFile = @"\13_Ragtime\Block1.SLDPRT";
        private const string SourceFile = @"C:\Work\Block1.SLDPRT";
        private const string Vault = FakePdmVault.VaultName;

        private static readonly string[] Writes =
        {
            "pdm.folder.create", "pdm.file.add", "pdm.file.checkout", "pdm.file.undo_checkout", "pdm.file.checkin", "pdm.variables.set",
        };

        private static readonly string[] OutsideTargets = { @"\Other\Block1.SLDPRT", @"\13_Ragtime2\Block1.SLDPRT", @"\" };

        private static readonly string[] MalformedTargets = { @"\13_Ragtime\..\Other\Block1.SLDPRT", @"13_Ragtime\Block1.SLDPRT", @"C:\Elsewhere\Block1.SLDPRT" };

        private static readonly string[] NonAsciiTargets = { "\\13_Ragtime\\Bl\u00f6ck.SLDPRT", "\\Other\\\u00dcber.SLDPRT" };

        private static readonly string[] FileInfoKeys =
        {
            "vault_path", "local_path", "name", "file_id", "exists_locally", "local_version", "latest_version", "checked_out",
            "checked_out_by", "checked_out_on", "checked_out_here", "checked_out_by_tool", "workflow_state", "in_approved_folder",
        };

        private static readonly string[] WriteMethods = { "CreateFolder", "AddFile", "CheckOut", "UndoCheckOut", "CheckIn", "SetVariables" };

        [Fact]
        public async Task FolderRightsAcceptsReadOnlyRootAndCollapsesDuplicates()
        {
            var h = new Harness();
            var arguments = Args(("paths", new JArray("\\", "/")));
            var result = Result(await h.CallAsync("pdm.folder.rights", arguments,
                TestData.LinkedContext(allowWrite: false, approved: new string[0])));
            var items = (JArray)result["items"]!;
            Assert.Single(items);
            Assert.Equal("\\", (string?)items[0]!["vault_path"]);
            Assert.True((bool)items[0]!["checked"]!);
        }

        [Fact]
        public async Task FolderRightsDistinguishesMissingAndFailedFolders()
        {
            var h = new Harness();
            h.Pdm.Vault.FolderRightsFailures[@"\Other"] = new PdmGatewayException(PdmFailureKind.Api, "failure");
            var result = Result(await h.CallAsync("pdm.folder.rights",
                Args(("paths", new JArray(Approved, @"\Missing", @"\Other"))),
                TestData.LinkedContext(allowWrite: false, approved: new string[0])));
            var items = (JArray)result["items"]!;
            Assert.True((bool)items[0]!["exists"]!);
            Assert.False((bool)items[1]!["exists"]!);
            Assert.True((bool)items[1]!["checked"]!);
            Assert.False((bool)items[2]!["checked"]!);
            Assert.Equal(JTokenType.Null, items[2]!["exists"]!.Type);
            Assert.Equal(JTokenType.Null, items[2]!["rights"]!.Type);
        }

        [Fact]
        public async Task FolderChildrenPagesSortedMembershipAndDetectsChanges()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFolder(@"\13_Ragtime\Z");
            h.Pdm.Vault.SeedFolder(@"\13_Ragtime\A");
            var context = TestData.LinkedContext(allowWrite: false, approved: new string[0]);
            var first = Result(await h.CallAsync("pdm.folder.children", Args(("path", S(Approved)), ("limit", I(1))), context));
            Assert.Equal(new[] { "A" }, Names(first));
            Assert.Equal(1, (int)first["next_offset"]!);
            Assert.Single(h.Pdm.Vault.Calls, call => call.StartsWith("GetFolderRights:", StringComparison.Ordinal));
            var snapshot = (string)first["snapshot"]!;
            h.Pdm.Vault.FolderRights[Approved + @"\Z"] = new PdmFolderRights { Read = true };
            var second = Result(await h.CallAsync("pdm.folder.children",
                Args(("path", S(Approved)), ("offset", I(1)), ("snapshot", S(snapshot))), context));
            Assert.Equal(new[] { "Z" }, Names(second));
            h.Pdm.Vault.SeedFolder(Approved + @"\B");
            AssertError(await h.CallAsync("pdm.folder.children",
                Args(("path", S(Approved)), ("snapshot", S(snapshot))), context), "stale_snapshot");
        }

        [Theory]
        [InlineData("pdm.folder.rights")]
        [InlineData("pdm.folder.children")]
        public async Task FolderDiscoveryRequiresALink(string operation)
        {
            var h = new Harness();
            var arguments = operation == "pdm.folder.rights"
                ? Args(("paths", new JArray("\\"))) : Args(("path", S("\\")));
            AssertError(await h.CallAsync(operation, arguments, TestData.Context()), "pdm_not_linked");
        }

        [Fact]
        public async Task FolderRightsRejectsEmptyAndOversizedLists()
        {
            var h = new Harness();
            var context = TestData.LinkedContext();
            AssertError(await h.CallAsync("pdm.folder.rights", Args(("paths", new JArray())), context), "invalid_arguments");
            AssertError(await h.CallAsync("pdm.folder.rights",
                Args(("paths", new JArray(Enumerable.Repeat("\\", 26)))), context), "invalid_arguments");
        }

        [Fact]
        public async Task FolderRightsHandlesLengthBoundsAndDeadline()
        {
            var h = new Harness();
            var longPath = "\\" + new string('a', 259);
            h.Pdm.Vault.SeedFolder(longPath);
            var context = TestData.LinkedContext();
            var result = Result(await h.CallAsync("pdm.folder.rights", Args(("paths", new JArray(longPath))), context));
            Assert.True((bool)result["items"]![0]!["exists"]!);
            AssertError(await h.CallAsync("pdm.folder.rights",
                Args(("paths", new JArray(longPath + "a"))), context), "invalid_arguments");
            h.Pdm.Vault.OnCall = (method, count) =>
            {
                if (method == "GetFolderRights") h.Clock.Advance(TimeSpan.FromSeconds(100));
            };
            result = Result(await h.CallAsync("pdm.folder.rights",
                Args(("paths", new JArray(Approved, @"\Other"))), context));
            Assert.True((bool)result["items"]![0]!["checked"]!);
            Assert.False((bool)result["items"]![1]!["checked"]!);
        }

        [Fact]
        public async Task FolderChildrenHandlesTimeoutMissingAndPastEnd()
        {
            var h = new Harness();
            var context = TestData.LinkedContext();
            AssertError(await h.CallAsync("pdm.folder.children", Args(("path", S(@"\Missing"))), context), "pdm_not_found");
            var empty = Result(await h.CallAsync("pdm.folder.children", Args(("path", S(Approved)), ("offset", I(100))), context));
            Assert.Empty((JArray)empty["items"]!);
            Assert.Equal(JTokenType.Null, empty["next_offset"]!.Type);
            h.Pdm.Vault.SeedFolder(Approved + @"\A");
            h.Pdm.Vault.OnCall = (method, count) =>
            {
                if (method == "ListSubFolderNames") h.Clock.Advance(TimeSpan.FromSeconds(100));
            };
            AssertError(await h.CallAsync("pdm.folder.children", Args(("path", S(Approved))), context), "operation_timeout");
        }

        [Fact]
        public async Task FolderChildrenTrimsPageToByteBudget()
        {
            var h = new Harness();
            for (var i = 0; i < 20; i++) h.Pdm.Vault.SeedFolder(Approved + "\\" + i.ToString("D2") + new string('x', 50));
            var context = TestData.LinkedContext();
            context["max_result_bytes"] = 1000;
            var result = Result(await h.CallAsync("pdm.folder.children", Args(("path", S(Approved))), context));
            var count = ((JArray)result["items"]!).Count;
            Assert.InRange(count, 1, 19);
            Assert.Equal(count, (int)result["next_offset"]!);
        }

        [Theory]
        [InlineData("pdm.folder.rights")]
        [InlineData("pdm.folder.children")]
        public async Task FolderDiscoveryDoesNotAccessClaims(string operation)
        {
            var h = new Harness();
            var file = h.Pdm.Vault.SeedFile(ApprovedFile);
            h.Claim(file);
            var claim = JObject.FromObject(h.Claims.Find(Vault, file.FileId)!);
            var findCount = h.Claims.FindCount;
            h.Claims.Calls.Clear();
            var arguments = operation == "pdm.folder.rights"
                ? Args(("paths", new JArray(Approved))) : Args(("path", S(Approved)));

            Result(await h.CallAsync(operation, arguments, TestData.LinkedContext()));

            Assert.Equal(findCount, h.Claims.FindCount);
            Assert.Empty(h.Claims.Calls);
            Assert.True(JToken.DeepEquals(claim, JObject.FromObject(h.Claims.Find(Vault, file.FileId)!)));
        }

        [Fact]
        public async Task FolderChildrenRejectsAnItemLargerThanTheByteBudget()
        {
            var h = new Harness();
            var name = new string('\u754c', 255);
            h.Pdm.Vault.SeedFolder(Approved + "\\" + name);
            var context = TestData.LinkedContext();
            context["max_result_bytes"] = 1000;

            AssertError(await h.CallAsync("pdm.folder.children", Args(("path", S(Approved))), context), "result_too_large");
        }

        [Fact]
        public async Task FolderChildrenSortsCaseInsensitiveTiesOrdinally()
        {
            var h = new Harness();
            h.Pdm.Vault.SubFolderNames = new[] { "abc", "Abd", "ABC" };
            h.Pdm.Vault.SeedFolder(Approved + @"\abc");
            h.Pdm.Vault.SeedFolder(Approved + @"\Abd");

            var result = Result(await h.CallAsync("pdm.folder.children", Args(("path", S(Approved))), TestData.LinkedContext()));

            Assert.Equal(new[] { "ABC", "abc", "Abd" }, Names(result));
        }

        [Fact]
        public async Task FolderChildrenKeepsOtherRightsWhenIndividualReadsFail()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFolder(Approved + @"\A");
            h.Pdm.Vault.SeedFolder(Approved + @"\B");
            h.Pdm.Vault.SeedFolder(Approved + @"\C");
            h.Pdm.Vault.FolderRightsFailures[Approved + @"\A"] = new PdmGatewayException(PdmFailureKind.Api, "failure");
            h.Pdm.Vault.FolderRightsFailures[Approved + @"\B"] = new PdmGatewayException(PdmFailureKind.NotFound, "missing");
            h.Pdm.Vault.FolderRights[Approved + @"\C"] = new PdmFolderRights { Read = true };

            var result = Result(await h.CallAsync("pdm.folder.children", Args(("path", S(Approved))), TestData.LinkedContext()));
            var items = (JArray)result["items"]!;

            Assert.Equal(new[] { "A", "B", "C" }, Names(result));
            Assert.Equal(JTokenType.Null, items[0]!["rights"]!.Type);
            Assert.Equal(JTokenType.Null, items[1]!["rights"]!.Type);
            Assert.True((bool)items[2]!["rights"]!["read"]!);
        }

        // ----- ladder: refusals for each write operation -----

        public static IEnumerable<object[]> WriteOperations() => Writes.Select(name => new object[] { name });

        public static IEnumerable<object[]> OutsideWrites() => Pairs(OutsideTargets);

        public static IEnumerable<object[]> MalformedWrites() => Pairs(MalformedTargets);

        public static IEnumerable<object[]> NonAsciiWrites() => Pairs(NonAsciiTargets);

        private static IEnumerable<object[]> Pairs(string[] targets) =>
            from operation in Writes
            from target in targets
            select new object[] { operation, target };

        [Theory]
        [MemberData(nameof(WriteOperations))]
        public async Task WriteRefusedWhenWritesAreNotAllowed(string operation)
        {
            var h = new Harness();
            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext(allowWrite: false));
            AssertError(response, "write_not_allowed");
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Theory]
        [MemberData(nameof(WriteOperations))]
        public async Task WriteRequiresItsRequiredArgument(string operation)
        {
            var h = new Harness();
            var arguments = ArgumentsFor(operation, ApprovedFile);
            arguments.Remove(RequiredField(operation));
            var response = await h.CallAsync(operation, arguments, TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal(RequiredField(operation), (string?)Details(response)["field"]);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Theory]
        [MemberData(nameof(WriteOperations))]
        public async Task WriteRequiresALinkedVault(string operation)
        {
            var h = new Harness();
            var context = TestData.LinkedContext();
            context.Remove("pdm");
            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), context);
            AssertError(response, "pdm_not_linked");
            Assert.Equal("no_link", (string?)Details(response)["reason"]);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Theory]
        [MemberData(nameof(MalformedWrites))]
        public async Task WriteRefusesMalformedPaths(string operation, string target)
        {
            var h = new Harness();
            var response = await h.CallAsync(operation, ArgumentsFor(operation, target), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal(0, h.Pdm.RunCount);
            Assert.Empty(WriteCalls(h));
        }

        [Theory]
        [MemberData(nameof(OutsideWrites))]
        public async Task WriteRefusesTargetsOutsideTheApprovedFolders(string operation, string target)
        {
            var h = new Harness();
            var response = await h.CallAsync(operation, ArgumentsFor(operation, target), TestData.LinkedContext());
            AssertError(response, "pdm_folder_not_approved");
            Assert.Equal(Approved, (string?)Details(response)["approved_folders"]![0]);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Theory]
        [MemberData(nameof(WriteOperations))]
        public async Task WriteRefusesEverythingWhenNoFolderIsApproved(string operation)
        {
            var h = new Harness();
            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext(approved: new string[0]));
            AssertError(response, "pdm_folder_not_approved");
            Assert.Empty((JArray)Details(response)["approved_folders"]!);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Theory]
        [MemberData(nameof(NonAsciiWrites))]
        public async Task WriteRefusesNonAsciiTargetsBeforeTheApprovalCheck(string operation, string target)
        {
            var h = new Harness();
            var response = await h.CallAsync(operation, ArgumentsFor(operation, target), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("Vault changes are limited to paths with plain ASCII characters.", (string?)response["error"]!["message"]);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Fact]
        public async Task FileAddRefusesNonAsciiNameBeforeTheApprovalCheck()
        {
            var h = new Harness();
            var arguments = Args(("source", S(SourceFile)), ("folder", S(Approved)), ("name", S("Bl\u00f6ck.SLDPRT")));
            var response = await h.CallAsync("pdm.file.add", arguments, TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("name", (string?)Details(response)["field"]);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Fact]
        public async Task FileAddRefusesADefaultedNameThatIsNotAValidName()
        {
            var h = new Harness();
            var response = await h.CallAsync("pdm.file.add", Args(("source", S(@"C:\Work\Block1.")), ("folder", S(Approved))), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("name", (string?)Details(response)["field"]);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Fact]
        public async Task ReadOperationsAcceptNonAsciiPaths()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile("\\13_Ragtime\\Bl\u00f6ck.SLDPRT");
            var info = Result(await h.CallAsync("pdm.file.info", Args(("path", S("\\13_Ragtime\\Bl\u00f6ck.SLDPRT"))), TestData.LinkedContext()));
            Assert.Equal("\\13_Ragtime\\Bl\u00f6ck.SLDPRT", (string?)info["vault_path"]);
            Assert.True((bool)info["in_approved_folder"]!);
        }

        // ----- status -----

        [Fact]
        public async Task StatusWithoutALinkReportsTheClientAndNoLinkedVault()
        {
            var h = new Harness();
            var response = await h.CallAsync("pdm.status", null, TestData.Context());
            var result = Result(response);
            AssertKeys(result, "client_installed", "client_version", "views", "linked", "work_folder");
            Assert.Equal(TestData.WorkFolder, (string?)result["work_folder"]);
            Assert.True((bool)result["client_installed"]!);
            Assert.Equal("32.3.1", (string?)result["client_version"]);
            var view = (JObject)result["views"]![0]!;
            AssertKeys(view, "vault", "root", "logged_in", "user");
            Assert.False((bool)view["logged_in"]!);
            Assert.Equal(JTokenType.Null, view["user"]!.Type);
            Assert.Equal(JTokenType.Null, result["linked"]!.Type);
            Assert.Equal(JTokenType.Null, response["host"]!["instance_id"]!.Type);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Fact]
        public async Task StatusWithALinkLogsInAndReportsTheVault()
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.status", null, TestData.LinkedContext()));
            var linked = (JObject)result["linked"]!;
            AssertKeys(linked, "vault", "root", "logged_in", "user", "writes_enabled", "approved_folders", "problem");
            Assert.Equal("PDM", (string?)linked["vault"]);
            Assert.Equal(@"C:\PDM", (string?)linked["root"]);
            Assert.True((bool)linked["logged_in"]!);
            Assert.Equal("mvisnovsky", (string?)linked["user"]);
            Assert.True((bool)linked["writes_enabled"]!);
            Assert.Equal(new[] { Approved }, linked["approved_folders"]!.ToObject<string[]>());
            Assert.Equal(JTokenType.Null, linked["problem"]!.Type);
            var view = (JObject)result["views"]![0]!;
            Assert.True((bool)view["logged_in"]!);
            Assert.Equal("mvisnovsky", (string?)view["user"]);
            Assert.Equal(new[] { Vault }, h.Pdm.RunVaults);
        }

        [Theory]
        [InlineData(true, new[] { "\\13_Ragtime" })]
        [InlineData(false, new[] { "\\13_Ragtime" })]
        [InlineData(true, new string[0])]
        public async Task StatusWritesEnabledNeedsWritesAndApprovedFolders(bool allowWrite, string[] approved)
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.status", null, TestData.LinkedContext(allowWrite: allowWrite, approved: approved)));
            Assert.Equal(allowWrite && approved.Length > 0, (bool)result["linked"]!["writes_enabled"]!);
        }

        [Theory]
        [InlineData("ClientMissing", "client_missing")]
        [InlineData("ViewMissing", "view_missing")]
        [InlineData("LoginFailed", "login_failed")]
        [InlineData("Busy", "busy")]
        public async Task StatusReportsAnUnavailableVaultWithoutFailing(string kind, string reason)
        {
            var h = new Harness();
            h.Pdm.RunFailure = new PdmGatewayException(ParseKind(kind), "Unavailable.");
            var linked = (JObject)Result(await h.CallAsync("pdm.status", null, TestData.LinkedContext()))["linked"]!;
            Assert.False((bool)linked["logged_in"]!);
            Assert.Equal(JTokenType.Null, linked["user"]!.Type);
            Assert.False((bool)linked["writes_enabled"]!);
            Assert.Equal(reason, (string?)linked["problem"]);
            Assert.Equal(@"C:\PDM", (string?)linked["root"]);
        }

        [Fact]
        public async Task StatusPropagatesATimeoutAsOperationTimeout()
        {
            var h = new Harness();
            h.Pdm.RunFailure = new PdmGatewayException(PdmFailureKind.Timeout, "Too slow.");
            AssertError(await h.CallAsync("pdm.status", null, TestData.LinkedContext()), "operation_timeout");
        }

        [Fact]
        public async Task StatusCarriesThePdmClientBlockWithoutLoggingIn()
        {
            var h = new Harness();
            var status = Result(await h.CallAsync("status", null, TestData.LinkedContext()));
            var pdm = (JObject)status["pdm"]!;
            AssertKeys(pdm, "client_installed", "client_version", "views");
            Assert.True((bool)pdm["client_installed"]!);
            Assert.Equal("32.3.1", (string?)pdm["client_version"]);
            Assert.Equal("PDM", (string?)pdm["views"]![0]!["vault"]);
            Assert.Empty(h.Pdm.RunVaults);
        }

        [Fact]
        public async Task StatusReportsAnAbsentClientAsNotInstalled()
        {
            var h = new Harness();
            h.Pdm.Facts = PdmClientFacts.NotInstalled;
            var status = Result(await h.CallAsync("status", null, TestData.Context()));
            Assert.False((bool)status["pdm"]!["client_installed"]!);
            Assert.Empty((JArray)status["pdm"]!["views"]!);
            Assert.Equal(JTokenType.Null, status["pdm"]!["client_version"]!.Type);
        }

        [Fact]
        public async Task ProcessorWithoutAGatewayTreatsThePdmClientAsMissing()
        {
            var linked = Result(await SendWithoutGateway("pdm.status", null, TestData.LinkedContext()))["linked"]!;
            Assert.Equal("client_missing", (string?)linked["problem"]);
            Assert.False((bool)linked["logged_in"]!);

            var status = Result(await SendWithoutGateway("status", null, TestData.Context()));
            Assert.False((bool)status["pdm"]!["client_installed"]!);

            var response = await SendWithoutGateway("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            AssertError(response, "pdm_unavailable");
            Assert.Equal("client_missing", (string?)Details(response)["reason"]);
        }

        // ----- folder list -----

        [Fact]
        public async Task FolderListOrdersFoldersFirstAndPagesWithSnapshots()
        {
            var h = new Harness();
            var vault = h.Pdm.Vault;
            vault.SeedFolder(@"\13_Ragtime\beta");
            vault.SeedFolder(@"\13_Ragtime\Alpha");
            vault.SeedFile(@"\13_Ragtime\b.SLDPRT");
            vault.SeedFile(@"\13_Ragtime\A.SLDPRT");
            vault.SeedFile(@"\13_Ragtime\c.SLDPRT");
            var context = TestData.LinkedContext();

            var first = Result(await h.CallAsync("pdm.folder.list", Args(("path", S(Approved)), ("limit", I(2))), context));
            Assert.Equal(5, (int)first["total"]!);
            Assert.Equal(2, (int)first["next_offset"]!);
            Assert.Equal(new[] { "Alpha", "beta" }, Names(first));
            var folder = (JObject)first["items"]![0]!;
            AssertKeys(folder, "name", "kind", "vault_path", "latest_version", "local_version", "checked_out");
            Assert.Equal("folder", (string?)folder["kind"]);
            Assert.Equal(@"\13_Ragtime\Alpha", (string?)folder["vault_path"]);
            Assert.Equal(JTokenType.Null, folder["latest_version"]!.Type);
            var snapshot = (string)first["snapshot"]!;

            var second = Result(await h.CallAsync("pdm.folder.list", Args(("path", S(Approved)), ("offset", I(2)), ("limit", I(2)), ("snapshot", S(snapshot))), context));
            Assert.Equal(new[] { "A.SLDPRT", "b.SLDPRT" }, Names(second));
            var file = (JObject)second["items"]![0]!;
            Assert.Equal("file", (string?)file["kind"]);
            Assert.Equal(3, (int)file["latest_version"]!);
            Assert.Equal(3, (int)file["local_version"]!);
            Assert.False((bool)file["checked_out"]!);
            Assert.Equal(4, (int)second["next_offset"]!);

            var last = Result(await h.CallAsync("pdm.folder.list", Args(("path", S(Approved)), ("offset", I(4)), ("limit", I(2)), ("snapshot", S(snapshot))), context));
            Assert.Equal(new[] { "c.SLDPRT" }, Names(last));
            Assert.Equal(JTokenType.Null, last["next_offset"]!.Type);

            vault.SeedFile(@"\13_Ragtime\d.SLDPRT");
            AssertError(await h.CallAsync("pdm.folder.list", Args(("path", S(Approved)), ("offset", I(2)), ("limit", I(2)), ("snapshot", S(snapshot))), context), "stale_snapshot");
        }

        [Fact]
        public async Task FolderListOfAMissingFolderIsNotFound()
        {
            var h = new Harness();
            var response = await h.CallAsync("pdm.folder.list", Args(("path", S(@"\13_Ragtime\Nope"))), TestData.LinkedContext());
            AssertError(response, "pdm_not_found");
            Assert.Equal(@"\13_Ragtime\Nope", (string?)Details(response)["vault_path"]);
        }

        // ----- file info, variables, latest -----

        [Fact]
        public async Task FileInfoReportsLiveStateAndApproval()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile, f =>
            {
                f.CheckedOut = true;
                f.CheckedOutBy = "other";
                f.CheckedOutOn = "2026-11-19 16:00";
            });
            var info = Result(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            AssertKeys(info, FileInfoKeys);
            Assert.Equal(ApprovedFile, (string?)info["vault_path"]);
            Assert.Equal(@"C:\PDM\13_Ragtime\Block1.SLDPRT", (string?)info["local_path"]);
            Assert.Equal("Block1.SLDPRT", (string?)info["name"]);
            Assert.Equal(3, (int)info["latest_version"]!);
            Assert.True((bool)info["in_approved_folder"]!);
            Assert.False((bool)info["checked_out_here"]!);
            Assert.False((bool)info["checked_out_by_tool"]!);
            Assert.Equal("other", (string?)info["checked_out_by"]);
            Assert.Equal(JTokenType.Null, info["workflow_state"]!.Type);
        }

        [Fact]
        public async Task FileInfoOutsideTheApprovedFoldersIsReadableAndMarkedNotApproved()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(@"\Other\Block1.SLDPRT");
            var info = Result(await h.CallAsync("pdm.file.info", Args(("path", S(@"\Other\Block1.SLDPRT"))), TestData.LinkedContext()));
            Assert.False((bool)info["in_approved_folder"]!);
        }

        [Fact]
        public async Task FileInfoNeverCheckedInReportsLatestVersionZero()
        {
            var h = new Harness();
            var added = Result(await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext()));
            Assert.Equal(0, (int)added["latest_version"]!);
            Assert.Equal(1, (int)added["local_version"]!);
        }

        [Fact]
        public async Task VariablesGetReturnsTheValuesOfTheRequestedConfiguration()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile);
            h.Pdm.Vault.ConfigurationVariables["Small"] = new Dictionary<string, string> { ["Description"] = "Small bracket" };
            var result = Result(await h.CallAsync("pdm.variables.get", Args(("path", S(ApprovedFile)), ("configuration", S("Small"))), TestData.LinkedContext()));
            AssertKeys(result, "vault_path", "configurations", "configuration", "variables");
            Assert.Equal(new[] { "Default", "Small" }, result["configurations"]!.ToObject<string[]>());
            Assert.Equal("Small", (string?)result["configuration"]);
            var variable = (JObject)result["variables"]![0]!;
            AssertKeys(variable, "name", "value");
            Assert.Equal("Small bracket", (string?)variable["value"]);
            Assert.Contains("GetVariables:" + ApprovedFile + "#Small", h.Pdm.Vault.Calls);
        }

        [Fact]
        public async Task GetLatestFetchesTheFileAndItsReferences()
        {
            var h = new Harness();
            var vault = h.Pdm.Vault;
            vault.SeedFile(ApprovedFile);
            vault.SeedFile(@"\13_Ragtime\Ref1.SLDPRT");
            vault.References.Add(@"\13_Ragtime\Ref1.SLDPRT");

            var result = Result(await h.CallAsync("pdm.file.get_latest", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            AssertKeys(result, "fetched", "skipped", "file");
            Assert.Equal(new[] { ApprovedFile, @"\13_Ragtime\Ref1.SLDPRT" }, result["fetched"]!.ToObject<string[]>());
            Assert.Empty((JArray)result["skipped"]!);
            Assert.Equal(ApprovedFile, (string?)result["file"]!["vault_path"]);

            var withoutReferences = Result(await h.CallAsync("pdm.file.get_latest", Args(("path", S(ApprovedFile)), ("include_references", new JValue(false))), TestData.LinkedContext()));
            Assert.Equal(new[] { ApprovedFile }, withoutReferences["fetched"]!.ToObject<string[]>());
        }

        [Fact]
        public async Task GetLatestSkipsAFileCheckedOutHereAndTooManyFilesIsRefused()
        {
            var h = new Harness();
            var vault = h.Pdm.Vault;
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var skipped = Result(await h.CallAsync("pdm.file.get_latest", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            var skip = (JObject)((JArray)skipped["skipped"]!)[0]!;
            Assert.Equal("checked_out_here", (string?)skip["reason"]);
            Assert.Empty((JArray)skipped["fetched"]!);
            Assert.True(file.CheckedOutHere);

            vault.Failures["GetLatest"] = new PdmGatewayException(PdmFailureKind.TooManyFiles, "Reference limit exceeded.");

            AssertError(await h.CallAsync("pdm.file.get_latest", Args(("path", S(ApprovedFile))), TestData.LinkedContext()), "result_too_large");
        }

        // ----- folder create -----

        [Fact]
        public async Task FolderCreateReportsAnExistingFolderWithoutCreatingIt()
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.folder.create", Args(("path", S(Approved))), TestData.LinkedContext()));
            AssertKeys(result, "vault_path", "local_path", "created");
            Assert.False((bool)result["created"]!);
            Assert.Empty(CreateCalls(h));
        }

        [Fact]
        public async Task FolderCreateMakesOneMissingLevel()
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.folder.create", Args(("path", S(@"C:\PDM\13_Ragtime\New"))), TestData.LinkedContext()));
            Assert.Equal(@"\13_Ragtime\New", (string?)result["vault_path"]);
            Assert.Equal(@"C:\PDM\13_Ragtime\New", (string?)result["local_path"]);
            Assert.True((bool)result["created"]!);
            Assert.Equal(new[] { @"CreateFolder:\13_Ragtime\New" }, CreateCalls(h));
        }

        [Fact]
        public async Task FolderCreateMakesTwoMissingLevelsTopDown()
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.folder.create", Args(("path", S(@"\13_Ragtime\A\B"))), TestData.LinkedContext()));
            Assert.True((bool)result["created"]!);
            Assert.Equal(new[] { @"CreateFolder:\13_Ragtime\A", @"CreateFolder:\13_Ragtime\A\B" }, CreateCalls(h));
        }

        [Fact]
        public async Task FolderCreateRefusesAMissingParentOutsideTheApprovedFolderAndCreatesNothing()
        {
            var h = new Harness();
            var response = await h.CallAsync("pdm.folder.create", Args(("path", S(@"\New\Sub\Deep"))), TestData.LinkedContext(approved: new[] { @"\New\Sub" }));
            AssertError(response, "pdm_folder_not_approved");
            Assert.Equal(@"\New", (string?)Details(response)["vault_path"]);
            Assert.Empty(CreateCalls(h));
            Assert.Equal(1, h.Pdm.RunCount);
        }

        [Fact]
        public async Task FolderCreateRefusesAFileAtThePath()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile);
            var response = await h.CallAsync("pdm.folder.create", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("already_exists", (string?)Details(response)["reason"]);
            Assert.Empty(CreateCalls(h));
        }

        [Fact]
        public async Task FolderCreateRefusesAPathThroughALink()
        {
            var h = new Harness();
            h.Pdm.Vault.LinkPaths.Add(@"\13_Ragtime\Junction");
            var response = await h.CallAsync("pdm.folder.create", Args(("path", S(@"\13_Ragtime\Junction\Sub"))), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("path", (string?)Details(response)["field"]);
            Assert.Empty(CreateCalls(h));
        }

        // ----- file add -----

        [Fact]
        public async Task FileAddUsesTheSourceFileNameByDefaultAndLeavesTheFileCheckedOut()
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext()));
            AssertKeys(result, FileInfoKeys);
            Assert.Equal(ApprovedFile, (string?)result["vault_path"]);
            Assert.Equal("Block1.SLDPRT", (string?)result["name"]);
            Assert.True((bool)result["checked_out"]!);
            Assert.True((bool)result["checked_out_here"]!);
            Assert.True((bool)result["checked_out_by_tool"]!);
            Assert.Equal(new[] { (SourceFile, Approved, "Block1.SLDPRT") }, h.Pdm.Vault.Added.ToArray());
        }

        [Fact]
        public async Task FileAddUsesAnExplicitName()
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(Approved)), ("name", S("Renamed.SLDPRT"))), TestData.LinkedContext()));
            Assert.Equal(@"\13_Ragtime\Renamed.SLDPRT", (string?)result["vault_path"]);
            Assert.Equal("Renamed.SLDPRT", h.Pdm.Vault.Added[0].Name);
        }

        [Fact]
        public async Task FileAddRefusesAnExistingTarget()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile);
            var response = await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("already_exists", (string?)Details(response)["reason"]);
            Assert.Empty(h.Pdm.Vault.Added);
        }

        [Fact]
        public async Task FileAddRefusesAMissingFolder()
        {
            var h = new Harness();
            var response = await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(@"\13_Ragtime\Nope"))), TestData.LinkedContext());
            AssertError(response, "pdm_not_found");
            Assert.Equal(@"\13_Ragtime\Nope", (string?)Details(response)["vault_path"]);
        }

        [Fact]
        public async Task FileAddAcceptsAnAbsoluteFolderUnderTheViewRoot()
        {
            var h = new Harness();
            var result = Result(await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(@"C:\PDM\13_Ragtime"))), TestData.LinkedContext()));
            Assert.Equal(ApprovedFile, (string?)result["vault_path"]);
            Assert.Equal(Approved, h.Pdm.Vault.Added[0].Folder);
        }

        [Fact]
        public async Task FileAddRefusesARelativeSource()
        {
            var h = new Harness();
            var response = await h.CallAsync("pdm.file.add", Args(("source", S("Block1.SLDPRT")), ("folder", S(Approved))), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("source", (string?)Details(response)["field"]);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Fact]
        public async Task FileAddReportsALocalFileInTheWayAsAConflictAndChangesNothing()
        {
            var h = new Harness();
            h.Pdm.Vault.Failures["AddFile"] = new PdmGatewayException(PdmFailureKind.LocalFileExists, "A file is in the way.", ApprovedFile);
            var response = await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("local_file_exists", (string?)Details(response)["reason"]);
            Assert.Empty(h.Pdm.Vault.Added);
            Assert.Null(h.Claims.Find(Vault, 100));
        }

        [Fact]
        public async Task FileAddMapsAnInvalidSourceFromTheGateway()
        {
            var h = new Harness();
            h.Pdm.Vault.Failures["AddFile"] = new PdmGatewayException(PdmFailureKind.InvalidSource, "The source is not a regular file.", SourceFile);
            var response = await h.CallAsync("pdm.file.add", Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("source", (string?)Details(response)["field"]);
            Assert.Equal("The source is not a regular file.", (string?)response["error"]!["message"]);
        }

        // ----- check-out, undo, check-in claims -----

        [Fact]
        public async Task CheckOutClaimsTheFileAndCheckInWithoutKeepingItReleasesTheClaim()
        {
            var h = new Harness();
            var file = h.Pdm.Vault.SeedFile(ApprovedFile, f => f.LocalVersion = 3);
            var checkedOut = Result(await h.CallAsync("pdm.file.checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.True((bool)checkedOut["checked_out_by_tool"]!);
            Assert.Equal(FakePdmVault.UserName, (string?)checkedOut["checked_out_by"]);
            Assert.NotNull(h.Claims.Find(Vault, file.FileId));

            var checkedIn = Result(await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)), ("comment", S("Rev B"))), TestData.LinkedContext()));
            Assert.False((bool)checkedIn["checked_out"]!);
            Assert.False((bool)checkedIn["checked_out_by_tool"]!);
            Assert.Equal(4, (int)checkedIn["latest_version"]!);
            Assert.Null(h.Claims.Find(Vault, file.FileId));
            Assert.Equal(CommentPrefixed("Rev B"), h.Pdm.Vault.CheckIns[0].Comment);
            Assert.False(h.Pdm.Vault.CheckIns[0].KeepCheckedOut);
        }

        [Fact]
        public async Task CheckInWithKeepCheckedOutKeepsTheClaim()
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var kept = Result(await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)), ("comment", S("Rev C")), ("keep_checked_out", new JValue(true))), TestData.LinkedContext()));
            Assert.True((bool)kept["checked_out_here"]!);
            Assert.True((bool)kept["checked_out_by_tool"]!);
            Assert.NotNull(h.Claims.Find(Vault, file.FileId));
            Assert.True(h.Pdm.Vault.CheckIns[0].KeepCheckedOut);
        }

        [Fact]
        public async Task UndoCheckOutOfAClaimedFileReleasesTheClaim()
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var undone = Result(await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.False((bool)undone["checked_out"]!);
            Assert.False((bool)undone["checked_out_here"]!);
            Assert.Null(h.Claims.Find(Vault, file.FileId));
        }

        [Theory]
        [InlineData("pdm.file.checkout")]
        [InlineData("pdm.file.undo_checkout")]
        [InlineData("pdm.file.checkin")]
        [InlineData("pdm.variables.set")]
        public async Task AFileCheckedOutHereButNotClaimedIsLeftAlone(string operation)
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: false);
            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("not_checked_out_by_tool", (string?)Details(response)["reason"]);
            Assert.Empty(WriteCalls(h));
        }

        [Fact]
        public async Task UndoConflictOrderIsNotHereThenNotClaimedThenNeverCheckedIn()
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: false, everCheckedIn: false);
            var unclaimed = await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            Assert.Equal("not_checked_out_by_tool", (string?)Details(unclaimed)["reason"]);

            var notHere = h.Pdm.Vault.SeedFile(@"\13_Ragtime\Other.SLDPRT", f =>
            {
                f.CheckedOut = true;
                f.CheckedOutBy = "other";
            });
            h.Claim(notHere);
            var response = await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(@"\13_Ragtime\Other.SLDPRT"))), TestData.LinkedContext());
            Assert.Equal("not_checked_out_here", (string?)Details(response)["reason"]);
        }

        [Fact]
        public async Task UndoOfAClaimedFileNeverCheckedInIsRefused()
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true, everCheckedIn: false);
            h.Claim(file, addedByTool: true);
            var response = await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("never_checked_in", (string?)Details(response)["reason"]);
            Assert.DoesNotContain(WriteCalls(h), call => call.StartsWith("UndoCheckOut", StringComparison.Ordinal));
        }

        [Fact]
        public async Task CheckOutOfAFileHeldByAnotherUserNamesTheHolder()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile, f =>
            {
                f.CheckedOut = true;
                f.CheckedOutBy = "other";
                f.CheckedOutOn = "2026-11-19 16:00";
            });
            var response = await h.CallAsync("pdm.file.checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("checked_out_by_other", (string?)Details(response)["reason"]);
            Assert.Equal("other", (string?)Details(response)["checked_out_by"]);
            Assert.Equal("2026-11-19 16:00", (string?)Details(response)["checked_out_on"]);
            Assert.DoesNotContain(WriteCalls(h), call => call.StartsWith("CheckOut", StringComparison.Ordinal));
        }

        [Fact]
        public async Task CheckOutOfAClaimedFileCheckedOutHereReturnsItUnchanged()
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: true);
            var result = Result(await h.CallAsync("pdm.file.checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.True((bool)result["checked_out_by_tool"]!);
            Assert.DoesNotContain(WriteCalls(h), call => call.StartsWith("CheckOut", StringComparison.Ordinal));
        }

        [Theory]
        [InlineData("Tab\tinside")]
        [InlineData("New\nline")]
        [InlineData("Del\u007F")]
        [InlineData("   ")]
        [InlineData("\tLeading tab")]
        public async Task CheckInRefusesControlCharactersAndWhitespaceOnlyComments(string comment)
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: true);
            var response = await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)), ("comment", S(comment))), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("comment", (string?)Details(response)["field"]);
            Assert.Empty(h.Pdm.Vault.CheckIns);
            Assert.Equal(0, h.Pdm.RunCount);
        }

        [Fact]
        public async Task CheckInTrimsTheCommentBeforeStoringIt()
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: true);
            Result(await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)), ("comment", S("  Rev B  "))), TestData.LinkedContext()));
            Assert.Equal(CommentPrefixed("Rev B"), h.Pdm.Vault.CheckIns[0].Comment);
        }

        [Fact]
        public async Task CheckInRefusesAFileNotCheckedOutHere()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile);
            var response = await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)), ("comment", S("Rev B"))), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("not_checked_out_here", (string?)Details(response)["reason"]);
            Assert.Empty(h.Pdm.Vault.CheckIns);
        }

        [Fact]
        public async Task UndoOfAFileNotCheckedOutHereIsRefused()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile);
            var response = await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            Assert.Equal("not_checked_out_here", (string?)Details(response)["reason"]);
        }

        // ----- variables -----

        [Fact]
        public async Task VariablesSetConvertsJsonValuesToTheirTypes()
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: true);
            var variables = new JObject
            {
                ["Description"] = S("Bracket B"),
                ["Count"] = I(3),
                ["Mass"] = new JValue(1.5),
                ["Flag"] = new JValue(true),
            };
            var result = Result(await h.CallAsync("pdm.variables.set", Args(("path", S(ApprovedFile)), ("variables", variables)), TestData.LinkedContext()));
            AssertKeys(result, "vault_path", "configuration", "updated");
            Assert.Equal("@", (string?)result["configuration"]);
            Assert.Equal(new[] { "Description", "Count", "Mass", "Flag" }, result["updated"]!.ToObject<string[]>());

            var written = h.Pdm.Vault.VariableWrites.Single();
            Assert.IsType<string>(written["Description"]);
            Assert.Equal(3L, Assert.IsType<long>(written["Count"]));
            Assert.Equal(1.5, Assert.IsType<double>(written["Mass"]));
            Assert.True(Assert.IsType<bool>(written["Flag"]));
        }

        [Fact]
        public async Task VariablesSetNamesAnUnknownVariableAndWritesNothing()
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: true);
            var response = await h.CallAsync("pdm.variables.set", Args(("path", S(ApprovedFile)), ("variables", new JObject { ["Nope"] = S("x") })), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("variables", (string?)Details(response)["field"]);
            Assert.Contains("Nope", (string?)response["error"]!["message"]);
            Assert.Empty(h.Pdm.Vault.VariableWrites);
        }

        // ----- mapping of gateway failures -----

        [Theory]
        [InlineData("ClientMissing", "client_missing")]
        [InlineData("ViewMissing", "view_missing")]
        [InlineData("LoginFailed", "login_failed")]
        [InlineData("Busy", "busy")]
        public async Task UnavailableKindsBecomePdmUnavailable(string kind, string reason)
        {
            var h = new Harness();
            h.Pdm.RunFailure = new PdmGatewayException(ParseKind(kind), "Unavailable.");
            var response = await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            AssertError(response, "pdm_unavailable");
            Assert.Equal(reason, (string?)Details(response)["reason"]);
            Assert.Equal(Vault, (string?)Details(response)["vault"]);
        }

        [Fact]
        public async Task TimeoutBecomesOperationTimeout()
        {
            var h = new Harness();
            h.Pdm.RunFailure = new PdmGatewayException(PdmFailureKind.Timeout, "Too slow.");
            AssertError(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext()), "operation_timeout");
        }

        [Fact]
        public async Task NotFoundBecomesPdmNotFound()
        {
            var h = new Harness();
            var response = await h.CallAsync("pdm.file.info", Args(("path", S(@"\13_Ragtime\Missing.SLDPRT"))), TestData.LinkedContext());
            AssertError(response, "pdm_not_found");
            Assert.Equal(@"\13_Ragtime\Missing.SLDPRT", (string?)Details(response)["vault_path"]);
        }

        [Fact]
        public async Task LinkInPathBecomesInvalidArgumentsOnPath()
        {
            var h = new Harness();
            h.Pdm.Vault.LinkPaths.Add(@"\13_Ragtime\Junction");
            var response = await h.CallAsync("pdm.file.checkout", Args(("path", S(@"\13_Ragtime\Junction\Block1.SLDPRT"))), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
            Assert.Equal("path", (string?)Details(response)["field"]);
            Assert.Equal("The path goes through a link (symbolic link or junction).", (string?)response["error"]!["message"]);
        }

        [Fact]
        public async Task UnknownVariableFromTheVaultIsInvalidArguments()
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: true);
            var response = await h.CallAsync("pdm.variables.set", Args(("path", S(ApprovedFile)), ("variables", new JObject { ["Nope"] = S("x") })), TestData.LinkedContext());
            AssertError(response, "invalid_arguments");
        }

        [Fact]
        public async Task ApiFailureBecomesPdmErrorWithHresultAndName()
        {
            var h = new Harness();
            h.Pdm.Vault.Failures["GetFile"] = new PdmGatewayException(PdmFailureKind.Api, "Access was denied.", null, unchecked((int)0x80004005), "E_EDM_PERMISSION_DENIED");
            var response = await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            AssertError(response, "pdm_error");
            Assert.Equal("0x80004005", (string?)Details(response)["hresult"]);
            Assert.Equal("E_EDM_PERMISSION_DENIED", (string?)Details(response)["pdm_error"]);
            Assert.Equal("Access was denied.", (string?)response["error"]!["message"]);
        }

        [Fact]
        public async Task ApiFailureWithoutAnHresultReportsNulls()
        {
            var h = new Harness();
            h.Pdm.Vault.Failures["GetFile"] = new PdmGatewayException(PdmFailureKind.Api, "Failed.");
            var response = await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            Assert.Equal(JTokenType.Null, Details(response)["hresult"]!.Type);
            Assert.Equal(JTokenType.Null, Details(response)["pdm_error"]!.Type);
        }

        // ----- isolation from SolidWorks -----

        [Fact]
        public async Task PdmRequestsNeverTouchSolidWorksInstancesOrTheMissingPinnedYear()
        {
            var h = new Harness();
            h.Pdm.Vault.SeedFile(ApprovedFile);
            h.Manager.Installed.Clear();
            h.Manager.Installed.Add(2024);
            var context = TestData.LinkedContext(version: 2030);

            var info = Result(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), context));
            var status = Result(await h.CallAsync("pdm.status", null, context));

            Assert.Equal(ApprovedFile, (string?)info["vault_path"]);
            Assert.True((bool)status["client_installed"]!);
            Assert.Empty(h.Manager.Targets);
            Assert.Empty(h.Manager.Deadlines);
        }

        [Theory]
        [InlineData("pdm.file.add")]
        [InlineData("pdm.file.checkout")]
        public async Task IncompleteClaimSurvivesAFailedFollowUpReadButRefusesLaterCheckIn(string operation)
        {
            var h = new Harness();
            if (operation == "pdm.file.checkout")
            {
                h.Pdm.Vault.SeedFile(ApprovedFile);
            }
            else
            {
                // The initial missing-file lookup is the first GetFile; the post-add lookup is the second.
                AssertError(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext()), "pdm_not_found");
            }

            h.Pdm.Vault.OnCall = (method, count) =>
            {
                if (method == "GetFile" && count == 2)
                {
                    throw new PdmGatewayException(PdmFailureKind.Api, "Follow-up read failed.");
                }
            };
            var arguments = operation == "pdm.file.add"
                ? Args(("source", S(SourceFile)), ("folder", S(Approved)))
                : Args(("path", S(ApprovedFile)));

            AssertError(await h.CallAsync(operation, arguments, TestData.LinkedContext()), "pdm_error");
            var claim = h.Claims.Find(Vault, 100);
            Assert.NotNull(claim);
            Assert.Equal(TestData.ToolId, claim!.ToolId);
            Assert.Equal(string.Empty, claim.LockPath);
            Assert.Single(WriteCalls(h));

            h.Pdm.Vault.OnCall = null;
            var checkedIn = await h.CallAsync("pdm.file.checkin",
                Args(("path", S(ApprovedFile)), ("comment", S("Finish successful mutation"))), TestData.LinkedContext());
            AssertError(checkedIn, "pdm_conflict");
            Assert.Equal("not_checked_out_by_tool", (string?)Details(checkedIn)["reason"]);
            Assert.Empty(h.Pdm.Vault.CheckIns);
            Assert.Null(h.Claims.Find(Vault, 100));
        }

        [Theory]
        [MemberData(nameof(WriteOperations))]
        public async Task AReadPassingTheDeadlinePreventsTheFollowingMutation(string operation)
        {
            var h = new Harness();
            CheckedOutHere(h, ApprovedFile, claimed: true);
            if (operation == "pdm.file.checkout")
            {
                h.Pdm.Vault.SeedFile(ApprovedFile);
            }

            var read = operation == "pdm.file.add" ? "FileExists"
                : operation == "pdm.folder.create" ? "FolderExists" : "GetFile";
            h.Pdm.Vault.OnCall = (method, count) =>
            {
                if (method == read)
                {
                    h.Clock.Now += 90000;
                }
            };
            var target = operation == "pdm.folder.create" ? Approved + @"\New" : ApprovedFile;
            var arguments = operation == "pdm.file.add"
                ? Args(("source", S(@"C:\Work\New.SLDPRT")), ("folder", S(Approved)))
                : ArgumentsFor(operation, target);

            AssertError(await h.CallAsync(operation, arguments, TestData.LinkedContext()), "operation_timeout");
            Assert.Empty(WriteCalls(h));
        }

        [Fact]
        public async Task FolderCreationStopsBetweenLevelsWhenTheDeadlineExpires()
        {
            var h = new Harness();
            h.Pdm.Vault.OnCall = (method, count) =>
            {
                if (method == "CreateFolder" && count == 1)
                {
                    h.Clock.Now += 90000;
                }
            };

            AssertError(await h.CallAsync("pdm.folder.create",
                Args(("path", S(Approved + @"\New\Child"))), TestData.LinkedContext()), "operation_timeout");
            Assert.Equal(new[] { "CreateFolder:" + Approved + @"\New" }, CreateCalls(h));
        }

        [Theory]
        [InlineData(260, true)]
        [InlineData(261, false)]
        public async Task FileAddValidatesTheCombinedTargetLengthBeforeApproval(int length, bool accepted)
        {
            var h = new Harness();
            var folder = Approved + "\\" + new string('a', 240 - Approved.Length - 1);
            var name = new string('b', length - folder.Length - 1);
            h.Pdm.Vault.SeedFolder(folder);
            var response = await h.CallAsync("pdm.file.add",
                Args(("source", S(SourceFile)), ("folder", S(folder)), ("name", S(name))), TestData.LinkedContext());

            if (accepted)
            {
                Assert.Equal(length, ((string)Result(response)["vault_path"]!).Length);
            }
            else
            {
                AssertError(response, "invalid_arguments");
                Assert.Equal("name", (string?)Details(response)["field"]);
                Assert.Equal(0, h.Pdm.RunCount);
            }
        }

        [Theory]
        [InlineData(@"C:\PDM\13_Ragtime\new.SLDPRT", null, true)]
        [InlineData(@"c:\pdm\13_ragtime\Sub\new.SLDPRT", null, true)]
        [InlineData(@"C:\PDM\13_Ragtime\.\new.SLDPRT", null, true)]
        [InlineData(@"C:\Work\new.SLDPRT", @"C:\Work", true)]
        [InlineData(@"C:\Work\Ü\new.SLDPRT", @"C:\Work", true)]
        [InlineData(@"c:\work\Sub\new.SLDPRT", @"C:\Work\", true)]
        [InlineData(@"C:\Work\new.SLDPRT", null, false)]
        [InlineData(@"C:\Work2\new.SLDPRT", @"C:\Work", false)]
        [InlineData(@"C:\PDM\13_Ragtime2\new.SLDPRT", @"C:\Work", false)]
        [InlineData(@"C:\Elsewhere\new.SLDPRT", @"C:\Work", false)]
        public async Task AddSourcesMustBeInsideAnApprovedViewFolderOrTheHostWorkFolder(string source, string? work, bool allowed)
        {
            var h = new Harness(workFolder: work);
            var response = await h.CallAsync("pdm.file.add",
                Args(("source", S(source)), ("folder", S(Approved)), ("name", S("new.SLDPRT"))), TestData.LinkedContext());
            if (allowed)
            {
                Result(response);
                Assert.Single(h.Pdm.Vault.Added);
            }
            else
            {
                AssertError(response, "invalid_arguments");
                Assert.Equal("source", (string?)Details(response)["field"]);
                Assert.Equal($"Add files from an approved vault folder or from the tool's work folder {work}.",
                    (string?)response["error"]!["message"]);
                Assert.Equal(0, h.Pdm.RunCount);
            }
        }

        [Fact]
        public async Task StatusIncludesNullableHostWorkFolder()
        {
            var h = new Harness(workFolder: null);
            var status = Result(await h.CallAsync("pdm.status"));
            Assert.Equal(JTokenType.Null, status["work_folder"]!.Type);
        }

        [Theory]
        [InlineData("pdm.file.checkout")]
        [InlineData("pdm.file.add")]
        public async Task SuccessfulMutationRecordsTheExactToolAndLockEvidence(string operation)
        {
            var h = new Harness();
            if (operation == "pdm.file.checkout") h.Pdm.Vault.SeedFile(ApprovedFile);
            var arguments = operation == "pdm.file.add"
                ? Args(("source", S(SourceFile)), ("folder", S(Approved)))
                : Args(("path", S(ApprovedFile)));
            var info = Result(await h.CallAsync(operation, arguments, TestData.LinkedContext()));
            var claim = h.Claims.Find(Vault, (int)info["file_id"]!);
            Assert.NotNull(claim);
            Assert.Equal(TestData.ToolId, claim!.ToolId);
            Assert.Equal(42, claim.LockedByUserId);
            Assert.Equal(FakePdmVault.CheckedOutOn, claim.LockedOnComputer);
            Assert.Equal(FakePdmVault.RootPath + ApprovedFile, claim.LockPath);
            Assert.Equal(h.UtcNow, claim.ClaimedUtc);
            Assert.True((bool)info["checked_out_by_tool"]!);
        }

        public static IEnumerable<object[]> InvalidClaims() =>
            from operation in new[] { "pdm.file.checkout", "pdm.file.checkin", "pdm.file.undo_checkout", "pdm.variables.set" }
            from change in new[] { "tool", "user", "computer", "path", "expired" }
            select new object[] { operation, change };

        [Theory]
        [MemberData(nameof(InvalidClaims))]
        public async Task InvalidClaimsAreRefusedButOnlyThisToolsStaleClaimsArePurged(string operation, string change)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            switch (change)
            {
                case "tool": h.Claim(file, "other-tool"); break;
                case "user": file.LockedByUserId = 43; break;
                case "computer": file.CheckedOutOn = "OTHER-PC"; break;
                case "path": file.LockPath = @"D:\Other\file.SLDPRT"; break;
                case "expired": h.UtcNow = h.UtcNow.AddHours(24); break;
            }
            var info = Result(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.False((bool)info["checked_out_by_tool"]!);

            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("not_checked_out_by_tool", (string?)Details(response)["reason"]);
            if (change == "tool")
            {
                Assert.Equal("other-tool", h.Claims.Find(Vault, file.FileId)!.ToolId);
            }
            else
            {
                Assert.Null(h.Claims.Find(Vault, file.FileId));
            }
            Assert.Empty(WriteCalls(h));
        }

        [Theory]
        [InlineData("pdm.file.checkin")]
        [InlineData("pdm.file.undo_checkout")]
        [InlineData("pdm.variables.set")]
        [InlineData("pdm.file.checkout")]
        public async Task AnotherToolsRefusedAttemptPreservesTheOwnersClaimAndTheOwnerCanCheckIn(string operation)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var original = h.Claims.Find(Vault, file.FileId);
            var otherContext = TestData.LinkedContext();
            otherContext["pdm"]!["tool_id"] = "tool-B";
            h.Claims.Calls.Clear();

            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), otherContext);

            AssertError(response, "pdm_conflict");
            Assert.Equal("not_checked_out_by_tool", (string?)Details(response)["reason"]);
            Assert.Equal("This file was checked out on this machine outside the tool; leave it to the person who did.",
                (string?)response["error"]!["message"]);
            Assert.Same(original, h.Claims.Find(Vault, file.FileId));
            Assert.Empty(h.Claims.Calls);
            Assert.Empty(WriteCalls(h));
            var otherInfo = Result(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), otherContext));
            Assert.False((bool)otherInfo["checked_out_by_tool"]!);
            var ownerInfo = Result(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.True((bool)ownerInfo["checked_out_by_tool"]!);

            Result(await h.CallAsync("pdm.file.checkin", ArgumentsFor("pdm.file.checkin", ApprovedFile), TestData.LinkedContext()));
            Assert.Single(h.Pdm.Vault.CheckIns);
            Assert.Null(h.Claims.Find(Vault, file.FileId));
        }

        [Theory]
        [InlineData("pdm.file.checkin")]
        [InlineData("pdm.file.undo_checkout")]
        [InlineData("pdm.variables.set")]
        [InlineData("pdm.file.checkout")]
        public async Task AnotherToolLeavesAnExpiredClaimForItsOwnerToPurge(string operation)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var original = h.Claims.Find(Vault, file.FileId);
            h.UtcNow = h.UtcNow.AddHours(24);
            var otherContext = TestData.LinkedContext();
            otherContext["pdm"]!["tool_id"] = "tool-B";
            h.Claims.Calls.Clear();

            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), otherContext);
            AssertError(response, "pdm_conflict");
            Assert.Equal("not_checked_out_by_tool", (string?)Details(response)["reason"]);
            Assert.Same(original, h.Claims.Find(Vault, file.FileId));
            Assert.Empty(h.Claims.Calls);

            response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("not_checked_out_by_tool", (string?)Details(response)["reason"]);
            Assert.Null(h.Claims.Find(Vault, file.FileId));
            Assert.Equal(new[] { "Remove" }, h.Claims.Calls);
            Assert.Empty(WriteCalls(h));
        }

        [Theory]
        [InlineData("pdm.file.info")]
        [InlineData("pdm.file.get_latest")]
        [InlineData("pdm.file.checkout")]
        [InlineData("pdm.file.checkin")]
        [InlineData("pdm.file.undo_checkout")]
        [InlineData("pdm.variables.set")]
        public async Task ObservingAFileNotCheckedOutHerePurgesItsClaim(string operation)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            file.CheckedOutHere = false;
            file.CheckedOut = true;
            await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext());
            Assert.Null(h.Claims.Find(Vault, file.FileId));
            Assert.Empty(WriteCalls(h));
        }

        [Fact]
        public async Task LockComputerAndPathComparisonIsOrdinalIgnoreCaseAndKeepCheckedOutRenewsTheClaim()
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            file.CheckedOutOn = "RenderPc";
            h.Claim(file);
            file.LockPath = file.LockPath!.ToUpperInvariant();
            file.CheckedOutOn = file.CheckedOutOn!.ToUpperInvariant();
            h.UtcNow = h.UtcNow.AddHours(23);
            var info = Result(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.True((bool)info["checked_out_by_tool"]!);
            Result(await h.CallAsync("pdm.file.checkin",
                Args(("path", S(ApprovedFile)), ("comment", S("Renew")), ("keep_checked_out", new JValue(true))), TestData.LinkedContext()));
            var renewed = h.Claims.Find(Vault, file.FileId)!;
            Assert.Equal(h.UtcNow, renewed.ClaimedUtc);
            Assert.Equal(file.LockPath, renewed.LockPath);
            Assert.Equal(file.CheckedOutOn, renewed.LockedOnComputer);
            h.UtcNow = h.UtcNow.AddHours(2);
            info = Result(await h.CallAsync("pdm.file.info", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.True((bool)info["checked_out_by_tool"]!);
        }

        [Fact]
        public async Task GetLatestReceivesTheLiveRequestDeadlineStopCallback()
        {
            var h = new Harness();
            var file = h.Pdm.Vault.SeedFile(ApprovedFile, f => f.LocalVersion = 1);
            h.Pdm.Vault.OnCall = (method, count) =>
            {
                if (method == "GetLatest") h.Clock.Now += 90000;
            };
            var result = Result(await h.CallAsync("pdm.file.get_latest", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.Empty((JArray)result["fetched"]!);
            Assert.Equal("deadline", (string?)result["skipped"]![0]!["reason"]);
            Assert.Equal(1, file.LocalVersion);
        }

        [Theory]
        [InlineData("pdm.file.checkin")]
        [InlineData("pdm.file.undo_checkout")]
        public async Task FailedDurableRevocationRefusesTheMutation(string operation)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var original = h.Claims.Find(Vault, file.FileId);
            h.Claims.RemoveFailure = new InvalidOperationException("Cannot persist removal.");
            h.Claims.Calls.Clear();

            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext());

            AssertError(response, "internal_error");
            Assert.Equal("The tool could not record the change safely; nothing was changed.",
                (string?)response["error"]!["message"]);
            Assert.Empty(WriteCalls(h));
            Assert.Same(original, h.Claims.Find(Vault, file.FileId));
            Assert.Equal(new[] { "Remove" }, h.Claims.Calls);
        }

        [Theory]
        [InlineData("pdm.file.checkin", "CheckIn")]
        [InlineData("pdm.file.undo_checkout", "UndoCheckOut")]
        public async Task MutationFailureRestoresTheSameRevokedClaim(string operation, string method)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var original = h.Claims.Find(Vault, file.FileId);
            h.Claims.Calls.Clear();
            h.Pdm.Vault.OnCall = (called, count) =>
            {
                if (called == method)
                {
                    Assert.Null(h.Claims.Find(Vault, file.FileId));
                    h.Claims.Calls.Add(called);
                }
            };
            h.Pdm.Vault.Failures[method] = new PdmGatewayException(PdmFailureKind.Api, "Mutation failed.");

            AssertError(await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext()), "pdm_error");

            Assert.Same(original, h.Claims.Find(Vault, file.FileId));
            Assert.Equal(new[] { "Remove", method, "Put" }, h.Claims.Calls);
            Assert.True(file.CheckedOutHere);
        }

        [Theory]
        [InlineData("pdm.file.checkin", "CheckIn")]
        [InlineData("pdm.file.undo_checkout", "UndoCheckOut")]
        public async Task SuccessfulMutationHasAlreadyDurablyRevokedTheClaim(string operation, string method)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            h.Claims.Calls.Clear();
            h.Pdm.Vault.OnCall = (called, count) =>
            {
                if (called == method)
                {
                    Assert.Null(h.Claims.Find(Vault, file.FileId));
                    h.Claims.Calls.Add(called);
                }
            };

            Result(await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext()));

            Assert.Null(h.Claims.Find(Vault, file.FileId));
            Assert.Equal(new[] { "Remove", method }, h.Claims.Calls);
        }

        [Fact]
        public async Task KeepCheckedOutStillMutatesBeforeRenewingAndNeverRevokes()
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var original = h.Claims.Find(Vault, file.FileId);
            h.Claims.RemoveFailure = new InvalidOperationException("Must not remove.");
            h.Claims.Calls.Clear();
            h.Pdm.Vault.OnCall = (called, count) =>
            {
                if (called == "CheckIn")
                {
                    Assert.Same(original, h.Claims.Find(Vault, file.FileId));
                    h.Claims.Calls.Add(called);
                }
            };
            h.UtcNow = h.UtcNow.AddHours(1);

            Result(await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)),
                ("comment", S("Keep working")), ("keep_checked_out", new JValue(true))), TestData.LinkedContext()));

            Assert.Equal(new[] { "CheckIn", "Put", "Put" }, h.Claims.Calls);
            Assert.Equal(h.UtcNow, h.Claims.Find(Vault, file.FileId)!.ClaimedUtc);
        }

        [Theory]
        [InlineData("pdm.file.checkin", "CheckIn")]
        [InlineData("pdm.file.undo_checkout", "UndoCheckOut")]
        public async Task FailedClaimRestorationIsLoggedAndDoesNotMaskTheMutationError(string operation, string method)
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var restoreFailure = new InvalidOperationException("Restore failed.");
            h.Claims.PutFailure = restoreFailure;
            h.Pdm.Vault.Failures[method] = new PdmGatewayException(PdmFailureKind.Api, "Mutation failed.");

            var response = await h.CallAsync(operation, ArgumentsFor(operation, ApprovedFile), TestData.LinkedContext());

            AssertError(response, "pdm_error");
            Assert.Equal("Mutation failed.", (string?)response["error"]!["message"]);
            Assert.Null(h.Claims.Find(Vault, file.FileId));
            Assert.Contains(restoreFailure, h.Unexpected);
            Assert.True(file.CheckedOutHere);
        }

        [Fact]
        public async Task AddedFileCannotBeUndoneBeforeCheckInEvenWithMisleadingGatewayHistory()
        {
            var h = new Harness();
            var added = Result(await h.CallAsync("pdm.file.add",
                Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext()));
            var id = (int)added["file_id"]!;
            Assert.True(h.Claims.Find(Vault, id)!.AddedByTool);
            // PDM may report this as checked in even though add has never been followed by check-in.
            h.Pdm.Vault.GetFile(ApprovedFile).EverCheckedIn = true;
            h.Pdm.Vault.Calls.Clear();
            h.Claims.Calls.Clear();

            var response = await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext());

            AssertError(response, "pdm_conflict");
            Assert.Equal("never_checked_in", (string?)Details(response)["reason"]);
            Assert.Empty(WriteCalls(h));
            Assert.Empty(h.Claims.Calls);
            Assert.True(h.Claims.Find(Vault, id)!.AddedByTool);
        }

        [Fact]
        public async Task CheckoutCanBeUndoneEvenWhenGatewayHistoryIsUnknown()
        {
            var h = new Harness();
            var file = h.Pdm.Vault.SeedFile(ApprovedFile, state => state.EverCheckedIn = false);
            Result(await h.CallAsync("pdm.file.checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            Assert.False(h.Claims.Find(Vault, file.FileId)!.AddedByTool);

            Result(await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));

            Assert.False(file.CheckedOutHere);
            Assert.Null(h.Claims.Find(Vault, file.FileId));
            Assert.Contains("UndoCheckOut:" + ApprovedFile, h.Pdm.Vault.Calls);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public async Task AddedFileCanBeUndoneAfterCheckInAndACurrentToolCheckout(bool keepCheckedOut)
        {
            var h = new Harness();
            var added = Result(await h.CallAsync("pdm.file.add",
                Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext()));
            var id = (int)added["file_id"]!;
            Result(await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)),
                ("comment", S("First check-in")), ("keep_checked_out", new JValue(keepCheckedOut))), TestData.LinkedContext()));
            if (!keepCheckedOut)
            {
                Assert.Null(h.Claims.Find(Vault, id));
                Result(await h.CallAsync("pdm.file.checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));
            }
            Assert.False(h.Claims.Find(Vault, id)!.AddedByTool);
            h.Pdm.Vault.GetFile(ApprovedFile).EverCheckedIn = false;

            Result(await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext()));

            Assert.Null(h.Claims.Find(Vault, id));
            Assert.Contains("UndoCheckOut:" + ApprovedFile, h.Pdm.Vault.Calls);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public async Task FailedFirstCheckInPreservesAddedProvenanceAndDoesNotRenewTheClaim(bool keepCheckedOut)
        {
            var h = new Harness();
            var added = Result(await h.CallAsync("pdm.file.add",
                Args(("source", S(SourceFile)), ("folder", S(Approved))), TestData.LinkedContext()));
            var id = (int)added["file_id"]!;
            var claimedUtc = h.Claims.Find(Vault, id)!.ClaimedUtc;
            h.UtcNow = h.UtcNow.AddHours(1);
            h.Pdm.Vault.Failures["CheckIn"] = new PdmGatewayException(PdmFailureKind.Api, "First check-in failed.");

            var response = await h.CallAsync("pdm.file.checkin", Args(("path", S(ApprovedFile)),
                ("comment", S("First check-in")), ("keep_checked_out", new JValue(keepCheckedOut))), TestData.LinkedContext());

            AssertError(response, "pdm_error");
            var claim = h.Claims.Find(Vault, id);
            Assert.NotNull(claim);
            Assert.True(claim!.AddedByTool);
            Assert.Equal(claimedUtc, claim.ClaimedUtc);
            h.Pdm.Vault.Calls.Clear();

            response = await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext());

            AssertError(response, "pdm_conflict");
            Assert.Equal("never_checked_in", (string?)Details(response)["reason"]);
            Assert.DoesNotContain("UndoCheckOut:" + ApprovedFile, h.Pdm.Vault.Calls);
            Assert.Empty(WriteCalls(h));
        }

        [Fact]
        public async Task LegacyClaimWithoutAddedByToolRefusesUndoButStillAllowsCheckIn()
        {
            var h = new Harness();
            var file = CheckedOutHere(h, ApprovedFile, claimed: true);
            var record = JObject.FromObject(h.Claims.Find(Vault, file.FileId)!);
            record.Remove("AddedByTool");
            h.Claims.Put(record.ToObject<PdmClaim>()!);
            Assert.True(h.Claims.Find(Vault, file.FileId)!.AddedByTool);

            var response = await h.CallAsync("pdm.file.undo_checkout", Args(("path", S(ApprovedFile))), TestData.LinkedContext());
            AssertError(response, "pdm_conflict");
            Assert.Equal("never_checked_in", (string?)Details(response)["reason"]);
            Assert.Empty(WriteCalls(h));

            Result(await h.CallAsync("pdm.file.checkin", ArgumentsFor("pdm.file.checkin", ApprovedFile), TestData.LinkedContext()));
            Assert.Null(h.Claims.Find(Vault, file.FileId));
        }

        // ----- helpers -----

        /// <summary>A file checked out on this machine; <paramref name="claimed"/> says whether the tool holds the claim.</summary>
        private static PdmFileState CheckedOutHere(Harness h, string path, bool claimed, bool everCheckedIn = true)
        {
            var file = h.Pdm.Vault.SeedFile(path, f =>
            {
                f.CheckedOut = true;
                f.CheckedOutBy = FakePdmVault.UserName;
                f.CheckedOutOn = FakePdmVault.CheckedOutOn;
                f.CheckedOutHere = true;
                f.LockedByUserId = 42;
                f.LockPath = f.LocalPath;
                f.EverCheckedIn = everCheckedIn;
            });
            if (claimed)
            {
                h.Claim(file);
            }

            return file;
        }

        private static string CommentPrefixed(string comment) => TestData.CommentPrefix + comment;

        private static PdmFailureKind ParseKind(string name) => (PdmFailureKind)Enum.Parse(typeof(PdmFailureKind), name);

        private static string RequiredField(string operation) => operation == "pdm.file.add" ? "source" : "path";

        /// <summary>Valid arguments for a write operation whose target is <paramref name="target"/> (a folder for file.add).</summary>
        private static JObject ArgumentsFor(string operation, string target)
        {
            switch (operation)
            {
                case "pdm.file.add":
                    return Args(("source", S(SourceFile)), ("folder", S(target)));
                case "pdm.file.checkin":
                    return Args(("path", S(target)), ("comment", S("Checked in by test")));
                case "pdm.variables.set":
                    return Args(("path", S(target)), ("variables", new JObject { ["Description"] = S("Bracket") }));
                default:
                    return Args(("path", S(target)));
            }
        }

        private static IEnumerable<string> WriteCalls(Harness h) =>
            h.Pdm.Vault.Calls.Where(call => WriteMethods.Any(method => call.StartsWith(method + ":", StringComparison.Ordinal)));

        private static string[] CreateCalls(Harness h) =>
            h.Pdm.Vault.Calls.Where(call => call.StartsWith("CreateFolder:", StringComparison.Ordinal)).ToArray();

        private static string[] Names(JObject page) =>
            ((JArray)page["items"]!).Select(item => item["name"]!.ToString()).ToArray();

        private static async Task<JObject> SendWithoutGateway(string operation, JObject? arguments, JObject context)
        {
            var processor = new RequestProcessor(
                new FakeInstanceManager(),
                new FakeHostFacts(),
                OperationRegistry.Default,
                TestData.HostVersion,
                ct => Task.CompletedTask,
                new FakeClock());
            return JObject.Parse(await processor.ProcessLineAsync(TestData.Request(operation, arguments, context), CancellationToken.None));
        }

        private static JObject Args(params (string Name, JToken Value)[] pairs) => Harness.Args(pairs);

        private static JToken S(string value) => Harness.S(value);

        private static JToken I(int value) => Harness.I(value);

        private static JObject Result(JObject response)
        {
            Assert.True((bool)response["ok"]!, response.ToString());
            return (JObject)response["result"]!;
        }

        private static void AssertError(JObject response, string code)
        {
            Assert.False((bool)response["ok"]!, response.ToString());
            Assert.Equal(code, (string?)response["error"]!["code"]);
        }

        private static JObject Details(JObject response) => (JObject)response["error"]!["details"]!;

        private static void AssertKeys(JObject value, params string[] keys) =>
            Assert.Equal(keys.OrderBy(key => key, StringComparer.Ordinal), value.Properties().Select(property => property.Name).OrderBy(key => key, StringComparer.Ordinal));
    }
}
