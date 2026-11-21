using System;
using System.IO;
using System.Text;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptWorkerFilesTests
    {
        [Theory]
        [InlineData("message", 2048)]
        [InlineData("stack", 8192)]
        [InlineData("exception_type", 2048)]
        public void FailureFieldsAreBounded(string field, int limit)
        {
            var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            try
            {
                ScriptWorkerFiles.WriteResult(directory, new JObject { ["status"] = "failed", [field] = new string('é', 1100000) });
                var result = ScriptWorkerFiles.ReadResult(directory);
                Assert.Equal("failed", (string?)result["status"]);
                Assert.True(Encoding.UTF8.GetByteCount((string)result[field]!) <= limit);
                Assert.True(new FileInfo(Path.Combine(directory, ScriptWorkerFiles.ResultName)).Length <= 1048576);
            }
            finally { Directory.Delete(directory, true); }
        }

        [Fact]
        public void OversizedEnvelopeIsReplacedByBoundedFailure()
        {
            var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            try
            {
                ScriptWorkerFiles.WriteResult(directory, new JObject { ["status"] = "completed", ["result"] = new string('x', 1048576) });
                var result = ScriptWorkerFiles.ReadResult(directory);
                Assert.Equal("failed", (string?)result["status"]);
                Assert.Equal("The script result was too large.", (string?)result["message"]);
                Assert.True(new FileInfo(Path.Combine(directory, ScriptWorkerFiles.ResultName)).Length <= 1048576);
                Assert.Single(Directory.GetFiles(directory));
            }
            finally { Directory.Delete(directory, true); }
        }

        [Fact]
        public void RequestAndResultRoundTripAtomically()
        {
            var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            try
            {
                var request = new JObject { ["run_id"] = "abc", ["arguments"] = new JObject { ["x"] = 1 } };
                ScriptWorkerFiles.WriteRequest(directory, request);
                Assert.True(JToken.DeepEquals(request, ScriptWorkerFiles.ReadRequest(directory)));
                var result = new JObject { ["status"] = "completed", ["result"] = 42 };
                ScriptWorkerFiles.WriteResult(directory, result);
                Assert.True(JToken.DeepEquals(result, ScriptWorkerFiles.ReadResult(directory)));
                Assert.Equal(2, Directory.GetFiles(directory).Length);
            }
            finally
            {
                Directory.Delete(directory, true);
            }
        }
    }
}
