using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    public class ArgumentValidationTests
    {
        public static IEnumerable<object[]> ValidArguments()
        {
            yield return new object[] { "script.run", "{\"source\":\"return null;\",\"purpose\":\"Inspect a model\",\"arguments\":{\"nested\":{\"items\":[null,1]}}}" };
            yield return new object[] { "describe", "{}" };
            yield return new object[] { "status", "{}" };
            yield return new object[] { "host.shutdown", "{}" };
            yield return new object[] { "host.shutdown", "{\"if_idle_seconds\":0}" };
            yield return new object[] { "host.shutdown", "{\"if_idle_seconds\":86400}" };
            yield return new object[] { "documents.list", "{}" };
            yield return new object[] { "document.open", "{\"path\":\"C:\\\\Models\\\\part.sldprt\"}" };
            yield return new object[] { "document.open", "{\"path\":\"C:\\\\Models\\\\a.SLDASM\",\"configuration\":\"Default\"}" };
            yield return new object[] { "document.close", "{\"document\":\"doc-1\"}" };
            yield return new object[] { "model.summary", "{\"document\":\"C:\\\\Models\\\\part.sldprt\"}" };
            yield return new object[] { "configurations.list", "{\"document\":\"doc-1\",\"offset\":0,\"limit\":500}" };
            yield return new object[] { "features.list", "{\"document\":\"doc-1\",\"include_system\":true,\"types\":[\"Extrusion\"]}" };
            yield return new object[] { "feature.get", "{\"document\":\"doc-1\",\"feature\":\"Boss-Extrude1\"}" };
            yield return new object[] { "sketch.get", "{\"document\":\"doc-1\",\"sketch\":\"Sketch1\",\"part\":\"segments\"}" };
            yield return new object[] { "sketch.get", "{\"document\":\"doc-1\",\"sketch\":\"Sketch1\"}" };
            yield return new object[] { "dimensions.list", "{\"document\":\"doc-1\",\"feature\":\"Boss-Extrude1\"}" };
            yield return new object[] { "mates.list", "{\"document\":\"doc-1\",\"snapshot\":\"" + new string('a', 64) + "\"}" };
            yield return new object[] { "components.list", "{\"document\":\"doc-1\",\"top_level_only\":false}" };
            yield return new object[] { "properties.get", "{\"document\":\"doc-1\",\"configuration\":\"Default\"}" };
            yield return new object[] { "mass_properties.get", "{\"document\":\"doc-1\"}" };
            yield return new object[] { "equations.list", "{\"document\":\"doc-1\"}" };
            yield return new object[] { "model.export", "{\"path\":\"C:\\\\Models\\\\Bracket.sldprt\",\"format\":\"step\"}" };
            yield return new object[] { "model.preview", "{\"path\":\"C:\\\\Models\\\\Bracket.sldprt\",\"view\":\"front\",\"width\":800}" };
            yield return new object[] { "pdm.status", "{}" };
            yield return new object[] { "pdm.folder.list", "{\"path\":\"\\\\A\"}" };
            yield return new object[] { "pdm.file.info", "{\"path\":\"\\\\A\"}" };
            yield return new object[] { "pdm.variables.get", "{\"path\":\"\\\\A\"}" };
            yield return new object[] { "pdm.folder.rights", "{\"paths\":[\"\\\\A\"]}" };
            yield return new object[] { "pdm.folder.children", "{\"path\":\"\\\\A\",\"limit\":100}" };
            yield return new object[] { "pdm.file.get_latest", "{\"path\":\"\\\\A\"}" };
            yield return new object[] { "pdm.folder.create", "{\"path\":\"\\\\A\"}" };
            yield return new object[] { "pdm.file.add", "{\"source\":\"C:\\\\a\",\"folder\":\"\\\\A\"}" };
            yield return new object[] { "pdm.file.checkout", "{\"path\":\"\\\\A\"}" };
            yield return new object[] { "pdm.file.undo_checkout", "{\"path\":\"\\\\A\"}" };
            yield return new object[] { "pdm.file.checkin", "{\"path\":\"\\\\A\",\"comment\":\"ok\"}" };
            yield return new object[] { "pdm.variables.set", "{\"path\":\"\\\\A\",\"variables\":{\"text\":\"x\",\"int\":1,\"float\":1.5,\"bool\":true}}" };
        }

        public static IEnumerable<object[]> InvalidArguments()
        {
            yield return new object[] { "script.run", "{}", "source" };
            yield return new object[] { "describe", "{\"x\":1}", "x" };
            yield return new object[] { "status", "{\"x\":1}", "x" };
            yield return new object[] { "host.shutdown", "{\"if_idle_seconds\":-1}", "if_idle_seconds" };
            yield return new object[] { "host.shutdown", "{\"if_idle_seconds\":86401}", "if_idle_seconds" };
            yield return new object[] { "host.shutdown", "{\"if_idle_seconds\":\"5\"}", "if_idle_seconds" };
            yield return new object[] { "host.shutdown", "{\"if_idle_seconds\":1.5}", "if_idle_seconds" };
            yield return new object[] { "host.shutdown", "{\"x\":1}", "x" };
            yield return new object[] { "documents.list", "{\"limit\":0}", "limit" };
            yield return new object[] { "document.open", "{}", "path" };
            yield return new object[] { "document.open", "{\"path\":\"C:\\\\a.sldprt\",\"bogus\":1}", "bogus" };
            yield return new object[] { "document.close", "{}", "document" };
            yield return new object[] { "model.summary", "{\"document\":\"\"}", "document" };
            yield return new object[] { "configurations.list", "{\"document\":\"doc-1\",\"offset\":-1}", "offset" };
            yield return new object[] { "features.list", "{\"document\":\"doc-1\",\"types\":[1]}", "types[0]" };
            yield return new object[] { "features.list", "{\"document\":\"doc-1\",\"include_system\":\"yes\"}", "include_system" };
            yield return new object[] { "feature.get", "{\"document\":\"doc-1\"}", "feature" };
            yield return new object[] { "sketch.get", "{\"document\":\"doc-1\",\"sketch\":\"S\",\"part\":\"bogus\"}", "part" };
            yield return new object[] { "sketch.get", "{\"document\":\"doc-1\",\"sketch\":\"S\",\"snapshot\":\"ABC\"}", "snapshot" };
            yield return new object[] { "dimensions.list", "{\"document\":\"doc-1\",\"feature\":5}", "feature" };
            yield return new object[] { "mates.list", "{\"document\":\"doc-1\",\"limit\":501}", "limit" };
            yield return new object[] { "components.list", "{\"document\":\"doc-1\",\"top_level_only\":\"no\"}", "top_level_only" };
            yield return new object[] { "properties.get", "{\"document\":\"doc-1\",\"configuration\":5}", "configuration" };
            yield return new object[] { "mass_properties.get", "{}", "document" };
            yield return new object[] { "equations.list", "{\"document\":\"doc-1\",\"limit\":\"ten\"}", "limit" };
            yield return new object[] { "model.export", "{\"path\":\"C:\\\\Models\\\\Bracket.sldprt\",\"format\":\"iges\"}", "format" };
            yield return new object[] { "model.preview", "{\"path\":\"C:\\\\Models\\\\Bracket.sldprt\",\"view\":\"back\"}", "view" };
            yield return new object[] { "features.list", "{\"document\":\"doc-1\",\"types\":[\"A\"],\"offset\":\"x\"}", "offset" };
            yield return new object[] { "pdm.status", "{\"x\":1}", "x" };
            yield return new object[] { "pdm.folder.list", "{}", "path" };
            yield return new object[] { "pdm.file.info", "{}", "path" };
            yield return new object[] { "pdm.variables.get", "{}", "path" };
            yield return new object[] { "pdm.folder.rights", "{}", "paths" };
            yield return new object[] { "pdm.folder.children", "{\"path\":\"\\\\A\",\"limit\":101}", "limit" };
            yield return new object[] { "pdm.file.get_latest", "{}", "path" };
            yield return new object[] { "pdm.folder.create", "{}", "path" };
            yield return new object[] { "pdm.file.add", "{}", "source" };
            yield return new object[] { "pdm.file.checkout", "{}", "path" };
            yield return new object[] { "pdm.file.undo_checkout", "{}", "path" };
            yield return new object[] { "pdm.file.checkin", "{\"path\":\"\\\\A\"}", "comment" };
            yield return new object[] { "pdm.variables.set", "{\"path\":\"\\\\A\",\"variables\":{}}", "variables" };
            yield return new object[] { "pdm.variables.set", "{\"path\":\"\\\\A\",\"variables\":{\"a\":null}}", "variables" };
            yield return new object[] { "pdm.variables.set", "{\"path\":\"\\\\A\",\"variables\":{\"a\":[]}}", "variables" };
        }

        [Theory]
        [MemberData(nameof(ValidArguments))]
        public void ValidArgumentsPass(string operation, string json)
        {
            var schema = OperationRegistry.Default.Find(operation)!.ArgumentsSchema;

            ArgumentValidator.Validate(schema, JObject.Parse(json));
        }

        [Theory]
        [MemberData(nameof(InvalidArguments))]
        public void InvalidArgumentsFailWithTheOffendingField(string operation, string json, string field)
        {
            var schema = OperationRegistry.Default.Find(operation)!.ArgumentsSchema;

            var error = Assert.Throws<HostErrorException>(() => ArgumentValidator.Validate(schema, JObject.Parse(json)));

            Assert.Equal(ErrorCodes.InvalidArguments, error.Code);
            Assert.Equal(field, (string?)error.Details["field"]);
        }

        [Fact]
        public void EveryPublicOperationHasValidAndInvalidCases()
        {
            var validOps = ValidArguments().Select(c => (string)c[0]).Distinct().ToHashSet();
            var invalidOps = InvalidArguments().Select(c => (string)c[0]).Distinct().ToHashSet();

            foreach (var operation in OperationRegistry.Default.Public)
            {
                Assert.Contains(operation.Name, validOps);
                Assert.Contains(operation.Name, invalidOps);
            }
        }

        [Fact]
        public void NullOptionalArgumentIsTreatedAsOmitted()
        {
            var schema = OperationRegistry.Default.Find("properties.get")!.ArgumentsSchema;

            ArgumentValidator.Validate(schema, JObject.Parse("{\"document\":\"doc-1\",\"configuration\":null}"));
        }

        [Fact]
        public void NullRequiredArgumentIsMissing()
        {
            var schema = OperationRegistry.Default.Find("feature.get")!.ArgumentsSchema;

            var error = Assert.Throws<HostErrorException>(() =>
                ArgumentValidator.Validate(schema, JObject.Parse("{\"document\":\"doc-1\",\"feature\":null}")));

            Assert.Equal("feature", (string?)error.Details["field"]);
        }

        [Fact]
        public void BooleanRejectsStrings()
        {
            var schema = OperationRegistry.Default.Find("components.list")!.ArgumentsSchema;

            Assert.Throws<HostErrorException>(() =>
                ArgumentValidator.Validate(schema, JObject.Parse("{\"document\":\"doc-1\",\"top_level_only\":\"true\"}")));
        }

        [Fact]
        public void ArrayLengthIsBounded()
        {
            var schema = OperationRegistry.Default.Find("features.list")!.ArgumentsSchema;
            var items = string.Join(",", Enumerable.Repeat("\"A\"", 51));

            var error = Assert.Throws<HostErrorException>(() =>
                ArgumentValidator.Validate(schema, JObject.Parse("{\"document\":\"doc-1\",\"types\":[" + items + "]}")));

            Assert.Equal("types", (string?)error.Details["field"]);
        }

        [Fact]
        public void UnsupportedSchemaShapeIsAProgrammingError()
        {
            var schema = JObject.Parse("{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\"}}}");

            Assert.Throws<InvalidOperationException>(() => ArgumentValidator.Validate(schema, JObject.Parse("{\"x\":1}")));
        }

        [Fact]
        public void IntegerBoundsAreInclusive()
        {
            var schema = OperationRegistry.Default.Find("documents.list")!.ArgumentsSchema;

            ArgumentValidator.Validate(schema, JObject.Parse("{\"limit\":1}"));
            ArgumentValidator.Validate(schema, JObject.Parse("{\"limit\":500}"));
            Assert.Throws<HostErrorException>(() => ArgumentValidator.Validate(schema, JObject.Parse("{\"limit\":1.5}")));
        }

        [Fact]
        public void PdmVariablesObjectAllowsAtMostFiftyProperties()
        {
            var schema = OperationRegistry.Default.Find("pdm.variables.set")!.ArgumentsSchema;
            var values = new JObject();
            for (var i = 0; i < 50; i++)
            {
                values.Add("v" + i, i);
            }

            var arguments = new JObject { ["path"] = "\\A", ["variables"] = values };
            ArgumentValidator.Validate(schema, arguments);
            values.Add("overflow", 0);
            var error = Assert.Throws<HostErrorException>(() => ArgumentValidator.Validate(schema, arguments));
            Assert.Equal("variables", (string?)error.Details["field"]);
        }
    }

    public class DescribeTests
    {
        private static readonly string[] PagedOperations =
        {
            "documents.list", "configurations.list", "features.list", "sketch.get",
            "dimensions.list", "mates.list", "components.list", "equations.list", "pdm.folder.list", "pdm.folder.children",
        };

        [Fact]
        public void DescribeHasEveryOperationWithReadAccess()
        {
            var result = DescribeBuilder.Build(OperationRegistry.Default, TestData.HostVersion);

            Assert.Equal(1, result.Protocol);
            Assert.Equal(TestData.HostVersion, result.HostVersion);
            Assert.Equal(32, result.Operations.Count);
            var writes = new[] { "pdm.folder.create", "pdm.file.add", "pdm.file.checkout", "pdm.file.undo_checkout", "pdm.file.checkin", "pdm.variables.set" };
            Assert.All(result.Operations, op => Assert.Equal(op.Name == "script.run" ? "script" : writes.Contains(op.Name) ? "write" : "read", op.Access));
            Assert.DoesNotContain(result.Operations, op => op.Name == "host.shutdown");
            Assert.All(new[] { "describe", "status", "document.open", "document.close", "model.summary", "feature.get", "mass_properties.get" },
                name => Assert.Contains(result.Operations, op => op.Name == name));
        }

        [Fact]
        public void EverySchemaIsAClosedObject()
        {
            var result = DescribeBuilder.Build(OperationRegistry.Default, TestData.HostVersion);

            foreach (var operation in result.Operations)
            {
                Assert.Equal("object", (string?)operation.Arguments["type"]);
                Assert.False((bool)operation.Arguments["additionalProperties"]!);
                Assert.NotNull(operation.Arguments["properties"]);
            }
        }

        [Fact]
        public void PagedOperationsAcceptPagingArgumentsAndOthersDoNot()
        {
            var result = DescribeBuilder.Build(OperationRegistry.Default, TestData.HostVersion);

            foreach (var operation in result.Operations)
            {
                var properties = (JObject)operation.Arguments["properties"]!;
                var accepts = properties.ContainsKey("offset") && properties.ContainsKey("limit") && properties.ContainsKey("snapshot");
                Assert.Equal(PagedOperations.Contains(operation.Name), operation.Paged);
                Assert.Equal(operation.Paged, accepts);
            }
        }

        [Fact]
        public void RequiredArgumentsAreListed()
        {
            var result = DescribeBuilder.Build(OperationRegistry.Default, TestData.HostVersion);
            var open = result.Operations.Single(op => op.Name == "document.open");

            Assert.Equal("path", (string?)open.Arguments["required"]![0]);
        }

        [Fact]
        public void DescribeFitsTheResultBudget()
        {
            var result = DescribeBuilder.Build(OperationRegistry.Default, TestData.HostVersion);

            var bytes = Encoding.UTF8.GetByteCount(WireJson.Compact(WireJson.ToToken(result)));

            Assert.True(bytes < ProtocolInfo.MaxResultBytes, $"describe is {bytes} bytes");
        }

        [Fact]
        public void DescribeIsGeneratedFromTheRegistry()
        {
            var registry = new OperationRegistry(new[]
            {
                new OperationDefinition("only.one", "Only.", "read", false, new ObjectSchema().Build(), "r"),
                new OperationDefinition("hidden", "Hidden.", "read", false, new ObjectSchema().Build(), "r", isInternal: true),
            });

            var result = DescribeBuilder.Build(registry, "9.9.9");

            Assert.Single(result.Operations);
            Assert.Equal("only.one", result.Operations[0].Name);
            Assert.Equal("9.9.9", result.HostVersion);
        }

        [Fact]
        public void DuplicateOperationNamesAreRejected()
        {
            var first = new OperationDefinition("dup", "a", "read", false, new ObjectSchema().Build(), "r");
            var second = new OperationDefinition("dup", "b", "read", false, new ObjectSchema().Build(), "r");

            Assert.Throws<ArgumentException>(() => new OperationRegistry(new[] { first, second }));
        }

        [Fact]
        public void LookupIsCaseSensitive()
        {
            Assert.NotNull(OperationRegistry.Default.Find("features.list"));
            Assert.Null(OperationRegistry.Default.Find("Features.List"));
            Assert.Null(OperationRegistry.Default.Find(null!));
        }

        [Fact]
        public async System.Threading.Tasks.Task DescribeOverTheProcessorListsTheSameOperations()
        {
            var h = new Harness();

            var response = await h.CallAsync("describe");

            Assert.Equal(32, ((JArray)response["result"]!["operations"]!).Count);
        }
    }
}
