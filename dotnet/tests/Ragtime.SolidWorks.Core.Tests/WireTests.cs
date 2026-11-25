using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class RequestParserTests
    {
        [Fact]
        public void ValidRequestParsesAllFields()
        {
            var line = TestData.Request("features.list", new JObject { ["document"] = "doc-1" }, TestData.Context(maxBytes: 1000));

            var parsed = RequestParser.Parse(line);

            Assert.Equal("req-1", parsed.Id);
            Assert.Equal("features.list", parsed.Operation);
            Assert.Equal("doc-1", (string?)parsed.Arguments["document"]);
            Assert.Equal(2024, parsed.Context.SolidworksVersion);
            Assert.Equal("background", parsed.Context.InstanceMode);
            Assert.Equal(1000, parsed.Context.MaxResultBytes);
            Assert.False(parsed.Context.AllowWrite);
            Assert.Equal(90, parsed.Context.DeadlineSeconds);
        }

        [Fact]
        public void NullArgumentsBecomeEmptyObject()
        {
            var line = "{\"protocol\":1,\"id\":\"x\",\"operation\":\"status\",\"arguments\":null,\"context\":" + TestData.Context().ToString() + "}";

            Assert.Empty(RequestParser.Parse(line).Arguments.Properties());
        }

        [Fact]
        public void NullSolidworksVersionIsAllowed()
        {
            var line = TestData.Request("status", null, TestData.Context(version: null));

            Assert.Null(RequestParser.Parse(line).Context.SolidworksVersion);
        }

        [Theory]
        [InlineData("")]
        [InlineData("   ")]
        [InlineData("not json")]
        [InlineData("[1,2]")]
        [InlineData("42")]
        public void MalformedInputIsInvalidRequest(string line)
        {
            var error = Assert.Throws<HostErrorException>(() => RequestParser.Parse(line));

            Assert.Equal(ErrorCodes.InvalidRequest, error.Code);
        }

        [Fact]
        public void MissingProtocolIsInvalidRequest()
        {
            var error = Assert.Throws<HostErrorException>(() =>
                RequestParser.Parse("{\"id\":\"x\",\"operation\":\"status\",\"context\":" + TestData.Context().ToString() + "}"));

            Assert.Equal(ErrorCodes.InvalidRequest, error.Code);
        }

        [Theory]
        [InlineData(2)]
        [InlineData(0)]
        [InlineData(99999999999)]
        public void OtherProtocolIsProtocolMismatch(long protocol)
        {
            var line = TestData.Request("status").Replace("\"protocol\":1", "\"protocol\":" + protocol);

            var error = Assert.Throws<HostErrorException>(() => RequestParser.Parse(line));

            Assert.Equal(ErrorCodes.ProtocolMismatch, error.Code);
            Assert.Equal(protocol, (long)error.Details["received"]!);
        }

        [Fact]
        public void StringProtocolIsInvalidRequestNotMismatch()
        {
            var line = TestData.Request("status").Replace("\"protocol\":1", "\"protocol\":\"1\"");

            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(line)).Code);
        }

        [Theory]
        [InlineData("\"id\":\"\"")]
        [InlineData("\"id\":5")]
        public void BadIdIsInvalidRequest(string idField)
        {
            var line = TestData.Request("status").Replace("\"id\":\"req-1\"", idField);

            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(line)).Code);
        }

        [Fact]
        public void MissingOperationIsInvalidRequest()
        {
            var line = TestData.Request("status").Replace("\"operation\":\"status\"", "\"operation\":\"\"");

            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(line)).Code);
        }

        [Fact]
        public void ArgumentsThatAreNotAnObjectAreInvalidRequest()
        {
            var line = "{\"protocol\":1,\"id\":\"x\",\"operation\":\"status\",\"arguments\":[1],\"context\":" + TestData.Context().ToString() + "}";

            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(line)).Code);
        }

        [Fact]
        public void MissingContextIsInvalidRequest()
        {
            var line = "{\"protocol\":1,\"id\":\"x\",\"operation\":\"status\"}";

            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(line)).Code);
        }

        [Theory]
        [InlineData("instance_mode", "\"fleet\"")]
        [InlineData("max_instances", "5")]
        [InlineData("max_instances", "0")]
        [InlineData("idle_shutdown_seconds", "59")]
        [InlineData("deadline_seconds", "0")]
        [InlineData("max_result_bytes", "32769")]
        [InlineData("solidworks_version", "2021")]
        [InlineData("allow_write", "\"yes\"")]
        public void OutOfRangeContextFieldIsInvalidRequest(string field, string value)
        {
            var context = TestData.Context();
            context[field] = JToken.Parse(value);
            var line = TestData.Request("status", null, context);

            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(line)).Code);
        }

        [Fact]
        public void RequestOverSixtyFourKibIsInvalidRequest()
        {
            var big = new string('a', ProtocolInfo.MaxRequestBytes);
            var line = TestData.Request("status", new JObject { ["filler"] = big });

            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(line)).Code);
        }

        [Fact]
        public void TryReadStringEchoesIdFromBrokenRequests()
        {
            Assert.Equal("abc", RequestParser.TryReadString("{\"id\":\"abc\",\"protocol\":\"x\"}", "id"));
            Assert.Null(RequestParser.TryReadString("{\"id\":5}", "id"));
            Assert.Null(RequestParser.TryReadString("garbage", "id"));
            Assert.Null(RequestParser.TryReadString(null, "id"));
        }
    }

    public class ResponseEnvelopeTests
    {
        [Fact]
        public void SuccessEnvelopeHasSnakeCaseKeysAndIsOneLine()
        {
            var host = new HostInfo("0.1.0", "32.5.0", 2024, "i1");
            var result = new JObject { ["text"] = WireJson.Value("line one\nline two") };

            var line = ResponseEnvelope.ToLine(ResponseEnvelope.Success("req-1", "features.list", result, host, 412));

            Assert.DoesNotContain("\n", line);
            Assert.DoesNotContain("\r", line);
            var root = JObject.Parse(line);
            Assert.Equal(1, (int)root["protocol"]!);
            Assert.Equal("req-1", (string?)root["id"]);
            Assert.True((bool)root["ok"]!);
            Assert.Equal("features.list", (string?)root["operation"]);
            Assert.Equal("line one\nline two", (string?)root["result"]!["text"]);
            Assert.Equal("0.1.0", (string?)root["host"]!["host_version"]);
            Assert.Equal("32.5.0", (string?)root["host"]!["solidworks_revision"]);
            Assert.Equal(2024, (int)root["host"]!["solidworks_version"]!);
            Assert.Equal("i1", (string?)root["host"]!["instance_id"]);
            Assert.Equal(412, (int)root["elapsed_ms"]!);
        }

        [Fact]
        public void NullHostFieldsAreWrittenAsNull()
        {
            var line = ResponseEnvelope.ToLine(ResponseEnvelope.Success("x", "describe", new JObject(), HostInfo.ForHost("0.1.0"), 1));

            var host = JObject.Parse(line)["host"]!;
            Assert.Equal(JTokenType.Null, host["solidworks_revision"]!.Type);
            Assert.Equal(JTokenType.Null, host["solidworks_version"]!.Type);
            Assert.Equal(JTokenType.Null, host["instance_id"]!.Type);
        }

        [Fact]
        public void FailureEnvelopeCarriesCodeMessageAndEmptyDetails()
        {
            var line = ResponseEnvelope.ToLine(ResponseEnvelope.Failure(
                "x", null, ErrorCodes.UnknownOperation, "No such operation.", null, HostInfo.ForHost("0.1.0"), 3));

            var root = JObject.Parse(line);
            Assert.False((bool)root["ok"]!);
            Assert.Equal(JTokenType.Null, root["operation"]!.Type);
            Assert.Equal("unknown_operation", (string?)root["error"]!["code"]);
            Assert.Equal("No such operation.", (string?)root["error"]!["message"]);
            Assert.Equal(JTokenType.Object, root["error"]!["details"]!.Type);
            Assert.False(root["error"]!["details"]!.HasValues);
            Assert.Null(root["result"]);
        }

        [Fact]
        public void FromExceptionUsesExceptionHostBlock()
        {
            var error = new HostErrorException(ErrorCodes.OperationTimeout, "Too slow.", null, new HostInfo("0.1.0", "32.5.0", 2024, "i2"));

            var root = ResponseEnvelope.FromException("x", "features.list", error, "0.1.0", 9);

            Assert.Equal("i2", (string?)root["host"]!["instance_id"]);
            Assert.Equal("operation_timeout", (string?)root["error"]!["code"]);
        }

        [Fact]
        public void FromExceptionWithoutHostUsesHostVersionOnly()
        {
            var root = ResponseEnvelope.FromException("x", "status", HostError.Coded(ErrorCodes.InternalError, "boom"), "0.1.0", 9);

            Assert.Equal("0.1.0", (string?)root["host"]!["host_version"]);
            Assert.Equal(JTokenType.Null, root["host"]!["instance_id"]!.Type);
        }

        [Fact]
        public void ClientHostNotRunningEnvelopeDescribesTheCause()
        {
            var line = ResponseEnvelope.ToLine(ClientEnvelopes.HostNotRunning("x", "status", "0.1.0", taskRegistered: true, interactiveSession: null, elapsedMs: 20000));

            var root = JObject.Parse(line);
            Assert.Equal("host_not_running", (string?)root["error"]!["code"]);
            Assert.True((bool)root["error"]!["details"]!["task_registered"]!);
            Assert.Equal(JTokenType.Null, root["error"]!["details"]!["interactive_session"]!.Type);
            Assert.Equal(20000, (int)root["elapsed_ms"]!);
        }
    }

    public class PagerTests
    {
        private static List<JObject> Items(int count, int padding = 0)
        {
            var items = new List<JObject>();
            for (var i = 0; i < count; i++)
            {
                items.Add(new JObject { ["index"] = WireJson.Value(i), ["pad"] = WireJson.Value(new string('p', padding)) });
            }

            return items;
        }

        private static PageRequest Request(int offset = 0, int limit = 100, string? snapshot = null) =>
            new PageRequest(offset, limit, snapshot);

        [Fact]
        public void FirstPageHasNextOffsetAndTotal()
        {
            var page = Pager.Page(Items(5), Request(limit: 2), "SI: m, kg, s, rad", 32768);

            Assert.Equal(2, ((JArray)page["items"]!).Count);
            Assert.Equal(0, (int)page["offset"]!);
            Assert.Equal(2, (int)page["next_offset"]!);
            Assert.Equal(5, (int)page["total"]!);
            Assert.Equal("SI: m, kg, s, rad", (string?)page["units"]);
        }

        [Fact]
        public void LastPageHasNullNextOffset()
        {
            var page = Pager.Page(Items(5), Request(offset: 4, limit: 2), "u", 32768);

            Assert.Single((JArray)page["items"]!);
            Assert.Equal(JTokenType.Null, page["next_offset"]!.Type);
        }

        [Fact]
        public void OffsetBeyondTotalReturnsEmptyPageWithNullNext()
        {
            var page = Pager.Page(Items(3), Request(offset: 50), "u", 32768);

            Assert.Empty((JArray)page["items"]!);
            Assert.Equal(JTokenType.Null, page["next_offset"]!.Type);
            Assert.Equal(3, (int)page["total"]!);
        }

        [Fact]
        public void EmptyListGivesEmptyPage()
        {
            var page = Pager.Page(Items(0), Request(), "u", 32768);

            Assert.Empty((JArray)page["items"]!);
            Assert.Equal(0, (int)page["total"]!);
            Assert.Equal(JTokenType.Null, page["next_offset"]!.Type);
        }

        [Fact]
        public void SnapshotIsSha256OfCompactItemArray()
        {
            var items = Items(3);
            var expectedText = "[" + string.Join(",", items.Select(i => i.ToString(Newtonsoft.Json.Formatting.None))) + "]";
            var expected = System.Security.Cryptography.SHA256.Create().ComputeHash(System.Text.Encoding.UTF8.GetBytes(expectedText));
            var expectedHex = string.Concat(expected.Select(b => b.ToString("x2")));

            var page = Pager.Page(items, Request(), "u", 32768);

            Assert.Equal(expectedHex, (string?)page["snapshot"]);
            Assert.Equal(64, ((string?)page["snapshot"])!.Length);
        }

        [Fact]
        public void SnapshotDoesNotDependOnPaging()
        {
            var first = Pager.Page(Items(10), Request(limit: 2), "u", 32768);
            var second = Pager.Page(Items(10), Request(offset: 4, limit: 3), "u", 32768);

            Assert.Equal((string?)first["snapshot"], (string?)second["snapshot"]);
        }

        [Fact]
        public void MatchingSnapshotIsAccepted()
        {
            var items = Items(4);
            var snapshot = (string?)Pager.Page(items, Request(), "u", 32768)["snapshot"];

            var page = Pager.Page(items, Request(offset: 2, snapshot: snapshot), "u", 32768);

            Assert.Equal(2, ((JArray)page["items"]!).Count);
        }

        [Fact]
        public void StaleSnapshotReturnsCurrentHashInDetails()
        {
            var before = Items(4);
            var snapshot = (string?)Pager.Page(before, Request(), "u", 32768)["snapshot"];
            var after = Items(5);
            var current = (string?)Pager.Page(after, Request(), "u", 32768)["snapshot"];

            var error = Assert.Throws<HostErrorException>(() => Pager.Page(after, Request(snapshot: snapshot), "u", 32768));

            Assert.Equal(ErrorCodes.StaleSnapshot, error.Code);
            Assert.Equal(current, (string?)error.Details["snapshot"]);
        }

        [Fact]
        public void PageNeverExceedsByteBudgetAndContinuesFromNextOffset()
        {
            var items = Items(40, padding: 300);
            const int budget = 4000;

            var collected = new List<int>();
            var offset = 0;
            int? next = 0;
            while (next.HasValue)
            {
                var page = Pager.Page(items, Request(offset: offset, limit: 500), "u", budget);
                Assert.True(WireJson.ByteCount(WireJson.Compact(page)) <= budget);
                foreach (var item in (JArray)page["items"]!)
                {
                    collected.Add((int)item["index"]!);
                }

                next = (int?)page["next_offset"];
                offset = next ?? offset;
            }

            Assert.Equal(Enumerable.Range(0, 40), collected);
        }

        [Fact]
        public void BudgetStopsBeforeLimitAndKeepsAtLeastOneItem()
        {
            var page = Pager.Page(Items(10, padding: 500), Request(limit: 500), "u", 1500);

            var count = ((JArray)page["items"]!).Count;
            Assert.True(count >= 1 && count < 10);
            Assert.Equal(count, (int)page["next_offset"]!);
        }

        [Fact]
        public void SingleItemOverBudgetIsResultTooLarge()
        {
            var error = Assert.Throws<HostErrorException>(() => Pager.Page(Items(2, padding: 5000), Request(), "u", 1000));

            Assert.Equal(ErrorCodes.ResultTooLarge, error.Code);
        }

        [Fact]
        public void NonPagedResultOverBudgetIsResultTooLarge()
        {
            var big = new JObject { ["text"] = WireJson.Value(new string('x', 2000)) };

            Assert.Throws<HostErrorException>(() => Pager.RequireFits(big, 1000));
            Assert.Same(big, Pager.RequireFits(big, 4000));
        }

        [Theory]
        [InlineData(-1)]
        [InlineData(501)]
        [InlineData(0)]
        public void LimitOutOfRangeIsInvalidArguments(int limit)
        {
            var args = new JObject { ["limit"] = WireJson.Value(limit) };

            Assert.Equal(ErrorCodes.InvalidArguments, Assert.Throws<HostErrorException>(() => PageRequest.FromArguments(args)).Code);
        }

        [Fact]
        public void PageRequestDefaultsAndSnapshotValidation()
        {
            var defaults = PageRequest.FromArguments(new JObject());
            Assert.Equal(0, defaults.Offset);
            Assert.Equal(100, defaults.Limit);
            Assert.Null(defaults.Snapshot);

            var badSnapshot = new JObject { ["snapshot"] = WireJson.Value("ABC") };
            Assert.Throws<HostErrorException>(() => PageRequest.FromArguments(badSnapshot));
        }
    }

    public class DocumentPathTests
    {
        [Theory]
        [InlineData(@"C:\Models\part.sldprt")]
        [InlineData(@"d:\models\ASM.SLDASM")]
        [InlineData(@"X:\drawing.slddrw")]
        [InlineData(@"C:\a b\c-d_e.SldPrt")]
        public void AcceptsAbsoluteLocalSolidWorksPaths(string path)
        {
            Assert.Null(DocumentPaths.FindProblem(path));
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData(@"\\server\share\part.sldprt")]
        [InlineData(@"\\?\C:\Models\part.sldprt")]
        [InlineData(@"\\.\C:\Models\part.sldprt")]
        [InlineData(@"Models\part.sldprt")]
        [InlineData(@"\Models\part.sldprt")]
        [InlineData(@"C:part.sldprt")]
        [InlineData("C:/Models/part.sldprt")]
        [InlineData(@"C:\Models\part.txt")]
        [InlineData(@"C:\Models\part")]
        [InlineData(@"C:\Models\part.sldprt.exe")]
        [InlineData(@"C:\Models\..\secret.sldprt")]
        [InlineData(@"C:\Models\\part.sldprt")]
        [InlineData(@"C:\Models\part.sldprt:stream.txt")]
        [InlineData(@"C:\Models\pa*rt.sldprt")]
        [InlineData("C:\\Models\\part\n.sldprt")]
        [InlineData("C:\\Models\\part\0.sldprt")]
        [InlineData("1:\\Models\\part.sldprt")]
        public void RejectsUnsafeOrUnsupportedPaths(string? path)
        {
            Assert.NotNull(DocumentPaths.FindProblem(path));
        }

        [Fact]
        public void RejectsPathsLongerThanLimit()
        {
            var path = @"C:\" + new string('a', DocumentPaths.MaxLength) + ".sldprt";

            Assert.NotNull(DocumentPaths.FindProblem(path));
        }

        [Fact]
        public void RequireOpenableThrowsInvalidArgumentsWithPathField()
        {
            var error = Assert.Throws<HostErrorException>(() => DocumentPaths.RequireOpenable(@"\\server\share\x.sldprt"));

            Assert.Equal(ErrorCodes.InvalidArguments, error.Code);
            Assert.Equal("path", (string?)error.Details["field"]);
        }

        [Fact]
        public void SamePathIgnoresCase()
        {
            Assert.True(DocumentPaths.SamePath(@"C:\A\Part.sldprt", @"c:\a\part.SLDPRT"));
            Assert.False(DocumentPaths.SamePath(@"C:\A\Part.sldprt", @"C:\A\Other.sldprt"));
        }
    }

    public class VersionTests
    {
        [Theory]
        [InlineData("32.5.0", 2024)]
        [InlineData("30.0.0", 2022)]
        [InlineData("33.1.0", 2025)]
        [InlineData("32", 2024)]
        public void RevisionMapsToReleaseYear(string revision, int year)
        {
            Assert.Equal(year, SolidWorksVersions.RevisionToYear(revision));
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("abc")]
        [InlineData("0.5.0")]
        [InlineData("-3.1")]
        public void UnreadableRevisionMapsToNull(string? revision)
        {
            Assert.Null(SolidWorksVersions.RevisionToYear(revision));
        }

        [Fact]
        public void YearAndMajorAreInverse()
        {
            Assert.Equal(2024, SolidWorksVersions.YearFromMajor(32));
            Assert.Equal(32, SolidWorksVersions.MajorFromYear(2024));
            Assert.Equal(2022, SolidWorksVersions.YearFromMajor(SolidWorksVersions.MajorFromYear(2022)));
        }

        [Theory]
        [InlineData(2021, false)]
        [InlineData(2022, true)]
        [InlineData(2035, true)]
        [InlineData(2036, false)]
        public void SupportedYearRange(int year, bool supported)
        {
            Assert.Equal(supported, SolidWorksVersions.IsSupportedYear(year));
        }

        [Theory]
        [InlineData(null, 2024, true)]
        [InlineData(null, null, true)]
        [InlineData(2024, 2024, true)]
        [InlineData(2024, 2025, false)]
        [InlineData(2024, null, false)]
        public void PinMatching(int? pinned, int? actual, bool matches)
        {
            Assert.Equal(matches, SolidWorksVersions.MatchesPin(pinned, actual));
        }
    }

    public class DeadlineTests
    {
        [Fact]
        public void RemainingCountsDownAndExpires()
        {
            var clock = new FakeClock { Now = 1000 };
            var deadline = new Deadline(10, clock);

            clock.Advance(TimeSpan.FromSeconds(4));
            Assert.Equal(TimeSpan.FromSeconds(6), deadline.Remaining);
            Assert.False(deadline.Expired);

            clock.Advance(TimeSpan.FromSeconds(7));
            Assert.Equal(TimeSpan.Zero, deadline.Remaining);
            Assert.True(deadline.Expired);
        }

        [Fact]
        public void DeadlineMustBeAtLeastOneSecond()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new Deadline(0, new FakeClock()));
        }

        [Fact]
        public void OverallCapIsDeadlinePlusFiveSeconds()
        {
            Assert.Equal(TimeSpan.FromSeconds(95), CallTiming.OverallCap(90));
        }

        [Fact]
        public void ForwardedDeadlineSubtractsSpentTimeRoundedDown()
        {
            Assert.Equal(88, CallTiming.ForwardedDeadlineSeconds(90, TimeSpan.FromMilliseconds(1500)));
            Assert.Equal(90, CallTiming.ForwardedDeadlineSeconds(90, TimeSpan.Zero));
            Assert.Equal(1, CallTiming.ForwardedDeadlineSeconds(5, TimeSpan.FromMilliseconds(3200)));
        }

        [Theory]
        [InlineData(5, 4001)]
        [InlineData(5, 5000)]
        [InlineData(5, 9000)]
        [InlineData(1, 999)]
        public void NothingLeftReturnsNull(int deadline, int spentMs)
        {
            Assert.Null(CallTiming.ForwardedDeadlineSeconds(deadline, TimeSpan.FromMilliseconds(spentMs)));
        }
    }

    public class CallRequestArgumentTests
    {
        [Fact]
        public void AcceptsTheLargestRequestRagtimeSends()
        {
            // MAX_ENCODED_SCRIPT_REQUEST_LENGTH in ragtime/solidworks/protocol.py: the largest script.run request.
            var value = new string('A', 32000);

            Assert.Equal(24000, CallRequestArgument.Decode(value).Length);
        }

        [Fact]
        public void LongerValueIsInvalidRequest()
        {
            var value = new string('A', CallRequestArgument.MaxLength + 4);

            var error = Assert.Throws<HostErrorException>(() => CallRequestArgument.Decode(value));

            Assert.Equal(ErrorCodes.InvalidRequest, error.Code);
        }

        [Fact]
        public void InvalidBase64IsInvalidRequest()
        {
            var error = Assert.Throws<HostErrorException>(() => CallRequestArgument.Decode("not base64!"));

            Assert.Equal(ErrorCodes.InvalidRequest, error.Code);
        }
    }

    public class HostNameTests
    {
        [Fact]
        public void NamesAreBuiltFromSid()
        {
            Assert.Equal("ragtime-solidworks-host-" + TestData.Sid, HostNames.PipeName(TestData.Sid));
            Assert.Equal("RagtimeSolidWorksHost-" + TestData.Sid, HostNames.TaskName(TestData.Sid));
            Assert.Equal(@"Global\ragtime-solidworks-host-" + TestData.Sid, HostNames.MutexName(TestData.Sid));
        }

        [Theory]
        [InlineData("")]
        [InlineData("administrator")]
        [InlineData("S-1-5-21-1; del")]
        public void InvalidSidIsRejected(string sid)
        {
            Assert.Throws<ArgumentException>(() => HostNames.PipeName(sid));
        }

        [Fact]
        public void RelativePathsMatchContract()
        {
            Assert.Equal(@"RagtimeSolidWorks\host", HostNames.InstallDirRelative);
            Assert.Equal(@"RagtimeSolidWorks\logs\host.log", HostNames.LogRelative);
            Assert.Equal(@"RagtimeSolidWorks\staging", HostNames.StagingRelative);
            Assert.Equal("ragtime-solidworks-host.exe", HostNames.ExeName);
        }
    }

    public class ContractConstantTests
    {
        [Fact]
        public void ErrorCodesMatchContractSectionFourExactly()
        {
            var expected = new[]
            {
                "invalid_request", "protocol_mismatch", "unknown_operation", "invalid_arguments", "write_not_allowed",
                "host_not_running", "no_interactive_session", "version_not_installed", "version_mismatch",
                "solidworks_start_failed", "solidworks_not_running", "instance_limit", "instance_timeout",
                "operation_timeout", "document_not_found", "document_not_open", "not_supported_for_document_type",
                "entity_not_found", "stale_snapshot", "result_too_large", "solidworks_error", "internal_error",
                "pdm_not_linked", "pdm_unavailable", "pdm_folder_not_approved", "pdm_not_found", "pdm_conflict", "pdm_error",
                "scripts_not_allowed", "script_compile_failed", "script_compile_unavailable", "script_invalid",
                "script_failed", "solidworks_unavailable", "host_busy", "host_stopping",
                "export_failed", "preview_unavailable",
            };

            Assert.Equal(expected, ErrorCodes.All);
        }

        [Fact]
        public void SizesAndVersionMatchContract()
        {
            Assert.Equal(1, ProtocolInfo.Version);
            Assert.Equal(32768, ProtocolInfo.MaxResultBytes);
            Assert.Equal(65536, ProtocolInfo.MaxRequestBytes);
        }
    }
}
