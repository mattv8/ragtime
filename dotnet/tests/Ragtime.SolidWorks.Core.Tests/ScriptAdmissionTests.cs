using System;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ScriptAdmissionTests
    {
        private static JObject Arguments() => new JObject { ["source"] = "return null;", ["purpose"] = "Inspect the model" };

        private static void Validate(JObject arguments) =>
            ArgumentValidator.Validate(OperationRegistry.Default.Find("script.run")!.ArgumentsSchema, arguments);

        private static void Invalid(JObject arguments, string field)
        {
            var error = Assert.Throws<HostErrorException>(() => Validate(arguments));
            Assert.Equal("invalid_arguments", error.Code);
            Assert.Equal(field, (string?)error.Details["field"]);
        }

        [Fact]
        public void CatalogListsScriptAccessAndExplicitSchemaLimits()
        {
            var definition = OperationRegistry.Default.Find("script.run")!;
            Assert.Equal("script", definition.Access);
            Assert.False(definition.Paged);
            Assert.False(definition.IsInternal);
            var schema = definition.ArgumentsSchema;
            Assert.Equal(new[] { "source", "purpose" }, schema["required"]!.ToObject<string[]>());
            var properties = schema["properties"]!;
            Assert.Equal(16384, (int)properties["source"]!["maxUtf8Bytes"]!);
            Assert.Equal(1, (int)properties["source"]!["minLength"]!);
            Assert.Equal(500, (int)properties["purpose"]!["maxLength"]!);
            Assert.Equal(4096, (int)properties["arguments"]!["maxJsonBytes"]!);
            Assert.Equal("object", (string?)properties["arguments"]!["type"]);
            Assert.Contains("600", definition.Result);
            Assert.Equal(64 * 1024, ProtocolInfo.MaxRequestBytes);
            Assert.Equal(32768, ProtocolInfo.MaxResultBytes);
        }

        [Theory]
        [InlineData("a", true)]
        [InlineData("", false)]
        [InlineData("é", true)]
        public void SourceMustHaveAtLeastOneUtf8Byte(string source, bool valid)
        {
            var arguments = Arguments(); arguments["source"] = source;
            if (valid) Validate(arguments);
            else Invalid(arguments, "source");
        }

        [Theory]
        [InlineData('a', 16384, true)]
        [InlineData('a', 16385, false)]
        [InlineData('é', 8192, true)]
        [InlineData('é', 8193, false)]
        public void SourceLimitCountsUtf8Bytes(char character, int count, bool valid)
        {
            var arguments = Arguments(); arguments["source"] = new string(character, count);
            if (valid) Validate(arguments);
            else Invalid(arguments, "source");
        }

        [Theory]
        [InlineData("")]
        [InlineData("purpose\n")]
        [InlineData("purpose\r")]
        [InlineData("purpose\t")]
        [InlineData("purpose\u007f")]
        [InlineData("\u0000purpose")]
        public void PurposeRejectsEmptyAndControlCharacters(string purpose)
        {
            var arguments = Arguments(); arguments["purpose"] = purpose;
            Invalid(arguments, "purpose");
        }

        [Fact]
        public void PurposeLimitIsInCharactersAndSourceMayContainNewlines()
        {
            var arguments = Arguments();
            arguments["purpose"] = new string('é', 500);
            arguments["source"] = "line 1\nline 2";
            Validate(arguments);
            arguments["purpose"] = new string('a', 501);
            Invalid(arguments, "purpose");
        }

        [Theory]
        [InlineData(2044, true)]
        [InlineData(2045, false)]
        public void ScriptArgumentsHaveACompactJsonUtf8Budget(int count, bool valid)
        {
            var arguments = Arguments();
            var payload = new JObject { ["x"] = new string('é', count) };
            Assert.Equal(8 + 2 * count, WireJson.ByteCount(WireJson.Compact(payload)));
            arguments["arguments"] = payload;
            if (valid) Validate(arguments);
            else Invalid(arguments, "arguments");
        }

        [Fact]
        public void ScriptArgumentsMayContainArbitraryNestedJsonButMustBeAnObject()
        {
            var arguments = Arguments();
            Validate(arguments);
            arguments["arguments"] = JObject.Parse("{'x':[1,true,null,{'key':'value'}]}");
            Validate(arguments);
            arguments["arguments"] = new JArray(1);
            Invalid(arguments, "arguments");
            arguments.Remove("arguments");
            arguments["unknown"] = true;
            Invalid(arguments, "unknown");
        }

        [Theory]
        [InlineData("i1", true)]
        [InlineData("a4242", true)]
        [InlineData("i0", false)]
        [InlineData("i1\n", false)]
        [InlineData("", false)]
        [InlineData("other", false)]
        public void InstanceIsOptionalAndUsesHostInstanceIdShape(string instance, bool valid)
        {
            var arguments = Arguments(); arguments["instance"] = instance;
            if (valid) Validate(arguments);
            else Invalid(arguments, "instance");
        }

        [Fact]
        public void MissingAllowScriptsDefaultsToFalseAndConstructorCallersRemainCompatible()
        {
            Assert.False(RequestParser.Parse(TestData.Request("script.run", Arguments())).Context.AllowScripts);
            Assert.False(TestData.DefaultContext().AllowScripts);
            var context = TestData.Context(); context["allow_scripts"] = true;
            Assert.True(RequestParser.Parse(TestData.Request("script.run", Arguments(), context)).Context.AllowScripts);
            context["allow_scripts"] = false;
            Assert.False(RequestParser.Parse(TestData.Request("script.run", Arguments(), context)).Context.AllowScripts);
        }

        [Theory]
        [InlineData("null")]
        [InlineData("1")]
        [InlineData("'true'")]
        public void ExplicitAllowScriptsMustBeBoolean(string json)
        {
            var context = TestData.Context(); context["allow_scripts"] = JToken.Parse(json);
            Assert.Equal("invalid_request", Assert.Throws<HostErrorException>(
                () => RequestParser.Parse(TestData.Request("script.run", Arguments(), context))).Code);
        }

        [Theory]
        [InlineData(false, false)]
        [InlineData(false, true)]
        [InlineData(true, false)]
        [InlineData(true, true)]
        public async Task ScriptsAndVaultWritesUseIndependentGates(bool writes, bool scripts)
        {
            var h = new Harness();
            var context = TestData.LinkedContext(allowWrite: writes);
            context["allow_scripts"] = scripts;
            var response = await h.CallAsync("script.run", Arguments(), context);
            Assert.Equal(scripts ? "internal_error" : "scripts_not_allowed", (string?)response["error"]!["code"]);
            Assert.Equal(scripts ? "Scripts are not available in this SolidWorks agent." : "Scripts are off for this tool. An admin can turn on Allow scripts in the tool's SolidWorks settings.",
                (string?)response["error"]!["message"]);
            Assert.Empty(h.Manager.Targets);

            var write = await h.CallAsync("pdm.folder.create", new JObject { ["path"] = @"\13_Ragtime" }, context);
            Assert.Equal(writes, (bool)write["ok"]!);
            if (!writes) Assert.Equal("write_not_allowed", (string?)write["error"]!["code"]);
        }

        [Fact]
        public async Task ScriptsGatePrecedesArgumentValidation()
        {
            var h = new Harness();
            var response = await h.CallAsync("script.run", new JObject());
            Assert.Equal("scripts_not_allowed", (string?)response["error"]!["code"]);
            var context = TestData.Context(); context["allow_scripts"] = true;
            response = await h.CallAsync("script.run", new JObject(), context);
            Assert.Equal("invalid_arguments", (string?)response["error"]!["code"]);
        }

        [Theory]
        [InlineData(600, "internal_error")]
        [InlineData(601, "invalid_arguments")]
        public async Task ScriptDeadlineCannotExceedSixHundredSeconds(int seconds, string code)
        {
            var h = new Harness();
            var context = TestData.Context(deadline: seconds); context["allow_scripts"] = true;
            var response = await h.CallAsync("script.run", Arguments(), context);
            Assert.Equal(code, (string?)response["error"]!["code"]);
            if (seconds > 600) Assert.Equal("timeout", (string?)response["error"]!["details"]!["field"]);
        }

        [Fact]
        public async Task ConfiguredRunnerExecutesScript()
        {
            var runner = new FakeScriptRunner();
            var h = new Harness(scriptRunner: runner);
            var context = TestData.Context(); context["allow_scripts"] = true;
            var response = await h.CallAsync("script.run", Arguments(), context);
            Assert.True((bool)response["ok"]!);
            Assert.Equal(1, runner.Calls);
        }

        [Fact]
        public async Task NewSeamMembersAreAvailableOnTheFakes()
        {
            ISolidWorksSession session = new FakeSession();
            Assert.Equal(4242, session.ProcessId);
            Assert.Equal(DateTimeKind.Utc, session.ProcessStartUtc.Kind);
            Assert.False(session.IsAttached);
            session.ReapplyWindowHiding();
            var manager = new FakeInstanceManager();
            Assert.Equal(ScriptRecovery.ProbeOk, await manager.RecoverAfterScriptAsync("i1", CancellationToken.None));
            Assert.Equal(new[] { "i1" }, manager.RecoveredInstances);
        }

        private sealed class FakeScriptRunner : IScriptRunner
        {
            public int Calls { get; private set; }
            public ScriptRunOutcome Run(ScriptRunRequest request, CancellationToken cancellationToken)
            {
                Calls++;
                return new ScriptRunOutcome { Status = ScriptRunStatus.Completed };
            }
        }
    }
}
