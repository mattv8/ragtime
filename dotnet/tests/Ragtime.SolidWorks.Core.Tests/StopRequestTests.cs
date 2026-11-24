using System.Linq;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class StopRequestTests
    {
        [Theory]
        [InlineData("0", 0)]
        [InlineData("5", 5)]
        [InlineData("007", 7)]
        [InlineData("86400", 86400)]
        public void ParsesDigitsInRange(string text, int expected)
        {
            Assert.Equal(expected, StopRequest.TryParseIfIdle(text));
        }

        [Theory]
        [InlineData("86401")]
        [InlineData("99999999999999999999")]
        [InlineData("-1")]
        [InlineData("+5")]
        [InlineData(" 5")]
        [InlineData("5 ")]
        [InlineData("")]
        [InlineData("1e3")]
        [InlineData("0x10")]
        [InlineData("\u0663")]
        [InlineData(null)]
        public void RejectsEverythingElse(string? text)
        {
            Assert.Null(StopRequest.TryParseIfIdle(text));
        }

        [Fact]
        public void AcceptedEnvelopeIsAccepted()
        {
            var reply = StopRequest.Interpret("{\"protocol\":1,\"id\":\"stop-1\",\"ok\":true,\"result\":{\"stopping\":true}}", "stop-1");

            Assert.Equal(StopReplyKind.Accepted, reply.Kind);
            Assert.Empty(reply.Reasons);
            Assert.Null(reply.Code);
        }

        [Fact]
        public void BusyEnvelopeKeepsOnlyStringReasons()
        {
            var line = "{\"protocol\":1,\"id\":\"stop-1\",\"ok\":false,\"error\":{\"code\":\"host_busy\",\"message\":\"The SolidWorks agent is in use.\",\"details\":{\"reasons\":[\"request_in_flight\",5,\"documents_open\",null]}}}";

            var reply = StopRequest.Interpret(line, "stop-1");

            Assert.Equal(StopReplyKind.Busy, reply.Kind);
            Assert.Equal(new[] { "request_in_flight", "documents_open" }, reply.Reasons.ToArray());
        }

        [Fact]
        public void BusyWithoutReasonsHasNoReasons()
        {
            var reply = StopRequest.Interpret("{\"id\":\"stop-1\",\"ok\":false,\"error\":{\"code\":\"host_busy\"}}", "stop-1");

            Assert.Equal(StopReplyKind.Busy, reply.Kind);
            Assert.Empty(reply.Reasons);
        }

        [Theory]
        [InlineData("{\"id\":\"stop-1\",\"ok\":false,\"error\":{\"code\":\"host_stopping\",\"message\":\"Stopping.\"}}", "host_stopping", "Stopping.")]
        [InlineData("{\"id\":\"stop-1\",\"ok\":false,\"error\":{\"code\":\"internal_error\"}}", "internal_error", "The SolidWorks agent reported an error.")]
        [InlineData("{\"id\":\"stop-1\",\"ok\":false,\"error\":{\"code\":7,\"message\":\"x\"}}", "internal_error", "x")]
        [InlineData("{\"id\":\"stop-1\",\"ok\":false,\"error\":{\"code\":\"pdm_error\",\"message\":3}}", "pdm_error", "The SolidWorks agent reported an error.")]
        [InlineData("{\"id\":\"stop-1\",\"ok\":false}", "internal_error", "The SolidWorks agent reported an error.")]
        public void OtherErrorsFailWithTheirCode(string line, string code, string message)
        {
            var reply = StopRequest.Interpret(line, "stop-1");

            Assert.Equal(StopReplyKind.Failed, reply.Kind);
            Assert.Equal(code, reply.Code);
            Assert.Equal(message, reply.Message);
            Assert.Empty(reply.Reasons);
        }

        [Theory]
        [InlineData("{\"id\":\"other\",\"ok\":true}")]
        [InlineData("{\"ok\":true}")]
        [InlineData("{\"id\":5,\"ok\":true}")]
        [InlineData("{\"id\":\"stop-1\",\"ok\":\"yes\"}")]
        [InlineData("{\"id\":\"stop-1\"}")]
        [InlineData("{\"id\":\"stop-1\",\"ok\":1}")]
        [InlineData("{\"id\":\"stop-1\", \"ok\":true")]
        [InlineData("not json")]
        [InlineData("[1,2]")]
        [InlineData("5")]
        [InlineData("")]
        [InlineData("   ")]
        [InlineData(null)]
        public void UnreadableAnswersFailWithInternalError(string? line)
        {
            var reply = StopRequest.Interpret(line, "stop-1");

            Assert.Equal(StopReplyKind.Failed, reply.Kind);
            Assert.Equal("internal_error", reply.Code);
            Assert.Equal("The SolidWorks agent gave an unreadable answer.", reply.Message);
        }
    }
}
