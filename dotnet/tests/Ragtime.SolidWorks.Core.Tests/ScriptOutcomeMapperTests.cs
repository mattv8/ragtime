using System;
using System.Numerics;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptOutcomeMapperTests
    {
        [Theory]
        [InlineData("completed", ScriptRunStatus.Completed)]
        [InlineData("compile_failed", ScriptRunStatus.CompileFailed)]
        [InlineData("compile_unavailable", ScriptRunStatus.CompileUnavailable)]
        [InlineData("invalid", ScriptRunStatus.Invalid)]
        [InlineData("failed", ScriptRunStatus.Failed)]
        [InlineData("attach_failed", ScriptRunStatus.AttachFailed)]
        public void StatusesMapToOutcomes(string status, ScriptRunStatus expected)
        {
            var result = new JObject { ["status"] = status, ["result"] = JValue.CreateNull() };

            var outcome = ScriptOutcomeMapper.FromResult(result, new[] { "x" });

            Assert.Equal(expected, outcome.Status);
        }

        [Fact]
        public void CompletedKeepsResultAndTimings()
        {
            var result = new JObject
            {
                ["status"] = "completed",
                ["result"] = new JObject { ["count"] = 3 },
                ["log_truncated"] = true,
                ["compile_ms"] = 120,
                ["run_ms"] = 45,
            };

            var outcome = ScriptOutcomeMapper.FromResult(result, new[] { "one", "two" });

            Assert.True(JToken.DeepEquals(new JObject { ["count"] = 3 }, outcome.Result));
            Assert.True(outcome.LogTruncated);
            Assert.Equal(120, outcome.CompileMilliseconds);
            Assert.Equal(45, outcome.RunMilliseconds);
            Assert.Equal(new[] { "one", "two" }, outcome.Log);
            Assert.Null(outcome.ExitCode);
        }

        [Fact]
        public void NullResultIsNotKept()
        {
            var outcome = ScriptOutcomeMapper.FromResult(
                new JObject { ["status"] = "completed", ["result"] = JValue.CreateNull() }, Array.Empty<string>());

            Assert.Null(outcome.Result);
        }

        [Fact]
        public void CompileErrorsKeepLinesAndHints()
        {
            var result = new JObject
            {
                ["status"] = "compile_failed",
                ["compile_errors"] = new JArray
                {
                    new JObject { ["line"] = 4, ["column"] = 9, ["code"] = "CS1056", ["message"] = "bad", ["hint"] = "C# 5 only; see the scripts guidance." },
                    new JObject { ["line"] = 7, ["column"] = 1, ["code"] = "CS0103", ["message"] = "missing", ["hint"] = null },
                },
            };

            var errors = ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()).CompileErrors;

            Assert.Equal(2, errors.Count);
            Assert.Equal(4, errors[0].Line);
            Assert.Equal(9, errors[0].Column);
            Assert.Equal("CS1056", errors[0].Code);
            Assert.Equal("C# 5 only; see the scripts guidance.", errors[0].Hint);
            Assert.Null(errors[1].Hint);
        }

        [Fact]
        public void FailureKeepsExceptionDetails()
        {
            var result = new JObject
            {
                ["status"] = "failed",
                ["exception_type"] = "InvalidOperationException",
                ["message"] = "boom",
                ["stack"] = "at RagtimeScript.Run",
            };

            var outcome = ScriptOutcomeMapper.FromResult(result, Array.Empty<string>());

            Assert.Equal("InvalidOperationException", outcome.ExceptionType);
            Assert.Equal("boom", outcome.Message);
            Assert.Equal("at RagtimeScript.Run", outcome.Stack);
        }

        [Fact]
        public void MalformedFieldCompileMsAsObjectIsRejected()
        {
            var result = new JObject { ["status"] = "completed", ["result"] = JValue.CreateNull(), ["compile_ms"] = new JObject() };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void MalformedUnknownStatusIsRejected()
        {
            var result = new JObject { ["status"] = "exploded" };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void MalformedMissingStatusIsRejected()
        {
            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(new JObject(), Array.Empty<string>()));
        }

        [Fact]
        public void MalformedCompileErrorsNotAnArrayIsRejected()
        {
            var result = new JObject { ["status"] = "compile_failed", ["compile_errors"] = "nope" };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void MalformedCompileErrorEntryNotAnObjectIsRejected()
        {
            var result = new JObject { ["status"] = "compile_failed", ["compile_errors"] = new JArray { 1 } };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void MalformedCompileErrorLineAsStringIsRejected()
        {
            var entry = new JObject { ["line"] = "4", ["column"] = 1, ["code"] = "CS1", ["message"] = "m" };
            var result = new JObject { ["status"] = "compile_failed", ["compile_errors"] = new JArray { entry } };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void MalformedCompletedWithoutResultFieldIsRejected()
        {
            var result = new JObject { ["status"] = "completed" };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void MalformedLogTruncatedAsStringIsRejected()
        {
            var result = new JObject { ["status"] = "failed", ["log_truncated"] = "yes" };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void MalformedMessageAsNumberIsRejected()
        {
            var result = new JObject { ["status"] = "failed", ["message"] = 7 };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Theory]
        [InlineData("9223372036854775808")]
        [InlineData("-9223372036854775809")]
        public void IntegerBeyondInt64IsRejectedForCompileMs(string value)
        {
            var result = new JObject
            {
                ["status"] = "completed",
                ["result"] = JValue.CreateNull(),
                ["compile_ms"] = new JValue(BigInteger.Parse(value)),
            };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Theory]
        [InlineData("line", "9223372036854775808")]
        [InlineData("line", "-9223372036854775809")]
        [InlineData("line", "3000000000")]
        [InlineData("column", "9223372036854775808")]
        [InlineData("column", "-9223372036854775809")]
        [InlineData("column", "-3000000000")]
        public void CompileErrorPositionOutsideInt32IsRejected(string field, string value)
        {
            var entry = new JObject { ["line"] = 1, ["column"] = 1, ["code"] = "CS1", ["message"] = "m" };
            entry[field] = new JValue(BigInteger.Parse(value));
            var result = new JObject { ["status"] = "compile_failed", ["compile_errors"] = new JArray { entry } };

            Assert.Throws<ScriptResultFormatException>(() => ScriptOutcomeMapper.FromResult(result, Array.Empty<string>()));
        }

        [Fact]
        public void NoResultReportsExitCode()
        {
            var outcome = ScriptOutcomeMapper.NoResult(2, new[] { "line" });

            Assert.Equal(ScriptRunStatus.Failed, outcome.Status);
            Assert.Equal(2, outcome.ExitCode);
            Assert.Equal(new[] { "line" }, outcome.Log);
        }

        [Fact]
        public void MalformedReportsFailedWithExitCodeAndReason()
        {
            var outcome = ScriptOutcomeMapper.Malformed(0, "\"compile_ms\" must be an integer.", new[] { "line" });

            Assert.Equal(ScriptRunStatus.Failed, outcome.Status);
            Assert.Equal(0, outcome.ExitCode);
            Assert.Contains("compile_ms", outcome.Message);
            Assert.Equal(new[] { "line" }, outcome.Log);
        }

        [Fact]
        public void TimedOutKeepsLog()
        {
            var outcome = ScriptOutcomeMapper.TimedOut(new[] { "before" });

            Assert.Equal(ScriptRunStatus.TimedOut, outcome.Status);
            Assert.Equal(new[] { "before" }, outcome.Log);
        }

        [Fact]
        public void ReadLogDecodesJsonStringLinesAndSkipsBadOnes()
        {
            var text = "\"first\"\n\n\"second \\\"quoted\\\"\"\n{not json\n\"[log truncated]\"\n";

            var lines = ScriptOutcomeMapper.ReadLog(text);

            Assert.Equal(new[] { "first", "second \"quoted\"", "[log truncated]" }, lines);
        }

        [Fact]
        public void ReadLogOfMissingTextIsEmpty()
        {
            Assert.Empty(ScriptOutcomeMapper.ReadLog(null));
            Assert.Empty(ScriptOutcomeMapper.ReadLog(string.Empty));
        }
    }
}
