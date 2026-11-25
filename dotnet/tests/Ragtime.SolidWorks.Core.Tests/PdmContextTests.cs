using System.Linq;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class PdmContextTests
    {
        private static JObject Request() => JObject.Parse(@"{'protocol':1,'id':'1','operation':'status','context':{'instance_mode':'background','max_instances':1,'idle_shutdown_seconds':60,'allow_write':false,'deadline_seconds':30,'max_result_bytes':32768}}");
        private static JObject Pdm() => new JObject { ["tool_id"] = TestData.ToolId, ["vault"] = "PDM", ["approved_folders"] = new JArray(@"\A"), ["comment_prefix"] = "" };

        [Fact]
        public void ParsesAbsentNullAndValidContext()
        {
            var request = Request();
            Assert.Null(RequestParser.Parse(request.ToString()).Context.Pdm);
            request["context"]!["pdm"] = JValue.CreateNull();
            Assert.Null(RequestParser.Parse(request.ToString()).Context.Pdm);
            request["context"]!["pdm"] = Pdm();
            var pdm = RequestParser.Parse(request.ToString()).Context.Pdm!;
            Assert.Equal("PDM", pdm.Vault);
            Assert.Equal(TestData.ToolId, pdm.ToolId);
            Assert.Equal(new[] { @"\A" }, pdm.ApprovedFolders);
            Assert.Equal("", pdm.CommentPrefix);
        }

        [Theory]
        [InlineData("vault", "")]
        [InlineData("vault", "A/B")]
        [InlineData("vault", "A\\B")]
        [InlineData("vault", "A\u007f")]
        [InlineData("comment_prefix", "A\u001f")]
        public void RejectsInvalidStrings(string field, string value)
        {
            var pdm = Pdm();
            pdm[field] = value;
            Reject(pdm);
        }

        [Fact]
        public void RejectsInvalidShapesAndBounds()
        {
            Reject(new JValue(1));
            foreach (var field in new[] { "vault", "approved_folders", "comment_prefix", "tool_id" })
            {
                var missing = Pdm(); missing.Remove(field); Reject(missing);
                var wrong = Pdm(); wrong[field] = 1; Reject(wrong);
            }
            var longVault = Pdm(); longVault["vault"] = new string('x', 129); Reject(longVault);
            var longPrefix = Pdm(); longPrefix["comment_prefix"] = new string('x', 161); Reject(longPrefix);
            var many = Pdm(); many["approved_folders"] = new JArray(Enumerable.Repeat(@"\A", 26)); Reject(many);
            foreach (var folder in new JToken[] { new JValue(1), new JValue(@"\"), new JValue(@"\A\"), new JValue("/A"), new JValue(@"C:\PDM\A"), new JValue(@"\A\..") })
            {
                var invalid = Pdm(); invalid["approved_folders"] = new JArray(folder); Reject(invalid);
            }
            var valid = Pdm(); valid["vault"] = new string('x', 128); valid["comment_prefix"] = new string('x', 160); valid["approved_folders"] = new JArray();
            var request = Request(); request["context"]!["pdm"] = valid;
            Assert.NotNull(RequestParser.Parse(request.ToString()).Context.Pdm);
            valid["approved_folders"] = new JArray(Enumerable.Repeat(@"\A", 25));
            Assert.Equal(25, RequestParser.Parse(request.ToString()).Context.Pdm!.ApprovedFolders.Count);
        }

        [Theory]
        [InlineData("\\13_Ränder")]
        [InlineData("\\13_Ragtime\\Ü")]
        [InlineData("\\13_Ragtime\\tab\t")]
        public void RejectsApprovedFoldersOutsidePlainAscii(string folder)
        {
            var pdm = Pdm();
            pdm["approved_folders"] = new JArray(folder);
            var request = Request(); request["context"]!["pdm"] = pdm;
            var error = Assert.Throws<HostErrorException>(() => RequestParser.Parse(request.ToString()));
            Assert.Equal(ErrorCodes.InvalidRequest, error.Code);
            Assert.Equal("Use plain ASCII characters in approved folder paths.", error.Message);
        }

        [Theory]
        [InlineData("")]
        [InlineData("a b")]
        [InlineData("a/b")]
        [InlineData("a\\b")]
        [InlineData("a.b")]
        [InlineData("ä")]
        [InlineData("a\n")]
        public void RejectsInvalidToolIds(string value)
        {
            var pdm = Pdm();
            pdm["tool_id"] = value;
            Reject(pdm);
        }

        [Fact]
        public void ToolIdHasExactAsciiAndLengthBounds()
        {
            foreach (var id in new[] { "a", "AZaz09_-", new string('a', 64) })
            {
                var request = Request();
                var pdm = Pdm(); pdm["tool_id"] = id; request["context"]!["pdm"] = pdm;
                Assert.Equal(id, RequestParser.Parse(request.ToString()).Context.Pdm!.ToolId);
            }
            var tooLong = Pdm(); tooLong["tool_id"] = new string('a', 65); Reject(tooLong);
        }

        private static void Reject(JToken pdm)
        {
            var request = Request(); request["context"]!["pdm"] = pdm;
            Assert.Equal(ErrorCodes.InvalidRequest, Assert.Throws<HostErrorException>(() => RequestParser.Parse(request.ToString())).Code);
        }
    }
}
