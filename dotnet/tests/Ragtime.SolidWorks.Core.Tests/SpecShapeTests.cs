using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Xunit;

namespace Ragtime.SolidWorks.Core.Tests
{
    /// <summary>
    /// Contract §5 shape: required members are always present (JSON null when there is no value); optional members
    /// are omitted when null. Checked per DTO on the wire form.
    /// </summary>
    public class SpecShapeSerializationTests
    {
        private static JObject Wire(object dto) => (JObject)WireJson.ToToken(dto);

        [Fact]
        public void PdmOperationsFollowContractOrderAndFlags()
        {
            var operations = OperationCatalog.Create().Where(operation => operation.Name.StartsWith("pdm.")).ToList();
            Assert.Equal(new[] { "pdm.status", "pdm.folder.list", "pdm.file.info", "pdm.variables.get",
                "pdm.folder.rights", "pdm.folder.children", "pdm.file.get_latest", "pdm.folder.create", "pdm.file.add", "pdm.file.checkout",
                "pdm.file.undo_checkout", "pdm.file.checkin", "pdm.variables.set" }, operations.Select(operation => operation.Name));
            for (var i = 0; i < operations.Count; i++)
            {
                Assert.Equal(i < 7 ? ProtocolInfo.Access.Read : ProtocolInfo.Access.Write, operations[i].Access);
                Assert.Equal(i == 1 || i == 5, operations[i].Paged);
            }
            var required = new[] { "", "path", "path", "path", "paths", "path", "path", "path", "source,folder",
                "path", "path", "path,comment", "path,variables" };
            for (var i = 0; i < operations.Count; i++)
            {
                var actual = (operations[i].ArgumentsSchema["required"] as JArray)?.Select(value => (string)value!).ToArray()
                    ?? new string[0];
                var expected = required[i].Split(new[] { ',' }, System.StringSplitOptions.RemoveEmptyEntries);
                Assert.Equal(expected, actual);
            }
        }

        [Fact]
        public void DocumentItemKeepsNullActiveConfigurationAndEmptyPath()
        {
            var root = Wire(new DocumentItem { DocumentId = "doc-1", InstanceId = "i1", Title = "Untitled1", Type = "part" });

            Assert.True(root.ContainsKey("active_configuration"));
            Assert.Equal(JTokenType.Null, root["active_configuration"]!.Type);
            Assert.Equal(string.Empty, (string?)root["path"]);
        }

        [Fact]
        public void PartWithoutMaterialHasNullMaterialAndConditionalCountersAbsent()
        {
            var root = Wire(new ModelSummary { DocumentId = "doc-1", Type = "part", Title = "Bracket", ConfigurationCount = 1, FeatureCount = 3, SolidBodyCount = 1 });

            Assert.True(root.ContainsKey("material"));
            Assert.Equal(JTokenType.Null, root["material"]!.Type);
            Assert.True(root.ContainsKey("active_configuration"));
            Assert.Equal(JTokenType.Null, root["active_configuration"]!.Type);
            Assert.True(root.ContainsKey("solid_body_count"));
            Assert.False(root.ContainsKey("component_count"));
            Assert.False(root.ContainsKey("mate_count"));
        }

        [Fact]
        public void AssemblyKeepsNullMaterialAndOmitsPartCounter()
        {
            var root = Wire(new ModelSummary { DocumentId = "doc-2", Type = "assembly", ComponentCount = 4, MateCount = 2 });

            Assert.True(root.ContainsKey("material"));
            Assert.Equal(JTokenType.Null, root["material"]!.Type);
            Assert.False(root.ContainsKey("solid_body_count"));
            Assert.True(root.ContainsKey("component_count"));
            Assert.True(root.ContainsKey("mate_count"));
        }

        [Fact]
        public void ModelSummaryWithMaterialSerializesIt()
        {
            var root = Wire(new ModelSummary { Type = "part", Material = "Steel", ActiveConfiguration = "Default" });

            Assert.Equal("Steel", (string?)root["material"]);
            Assert.Equal("Default", (string?)root["active_configuration"]);
        }

        [Fact]
        public void RootConfigurationHasNullParentAndDescriptionPresent()
        {
            var root = Wire(new ConfigurationItem { Name = "Default", IsActive = true });

            Assert.True(root.ContainsKey("parent"));
            Assert.Equal(JTokenType.Null, root["parent"]!.Type);
            Assert.True(root.ContainsKey("description"));
            Assert.Equal(JTokenType.Null, root["description"]!.Type);
        }

        [Fact]
        public void ChildConfigurationSerializesItsParent()
        {
            var root = Wire(new ConfigurationItem { Name = "Small", Parent = "Default", Description = "Small variant" });

            Assert.Equal("Default", (string?)root["parent"]);
            Assert.Equal("Small variant", (string?)root["description"]);
        }

        [Fact]
        public void HealthyFeatureDetailHasNullWhatsWrongAndOmitsSketch()
        {
            var root = Wire(new FeatureDetail { Name = "Boss-Extrude1", Type = "Extrusion", Id = "1" });

            Assert.True(root.ContainsKey("whats_wrong"));
            Assert.Equal(JTokenType.Null, root["whats_wrong"]!.Type);
            Assert.Equal("0", (string?)root["error_code"]);
            Assert.False(root.ContainsKey("sketch"));
        }

        [Fact]
        public void FeatureDetailWithSketchSerializesIt()
        {
            var root = Wire(new FeatureDetail { Name = "Boss-Extrude1", Type = "Extrusion", Id = "1", Sketch = "Sketch1" });

            Assert.Equal("Sketch1", (string?)root["sketch"]);
        }

        [Fact]
        public void EquationWithoutFiniteValueHasNullValue()
        {
            var root = Wire(new EquationItem { Index = 0, Equation = "\"D1\"=\"x\"" });

            Assert.True(root.ContainsKey("value"));
            Assert.Equal(JTokenType.Null, root["value"]!.Type);
        }

        [Fact]
        public void DimensionWithoutFeatureHasNullFeatureAndOmitsTolerance()
        {
            var root = Wire(new DimensionItem { Name = "D2", Value = 0.5, Type = "angular" });

            Assert.True(root.ContainsKey("feature"));
            Assert.Equal(JTokenType.Null, root["feature"]!.Type);
            Assert.False(root.ContainsKey("tolerance"));
        }

        [Fact]
        public void SketchSummaryHasNullPlaneWhenNotKnown()
        {
            var root = Wire(new SketchSummary { Name = "Sketch1", Units = ProtocolInfo.SiUnits });

            Assert.True(root.ContainsKey("plane_or_face"));
            Assert.Equal(JTokenType.Null, root["plane_or_face"]!.Type);
        }

        [Fact]
        public void TopLevelComponentHasNullParentAndPresentPathAndConfiguration()
        {
            var root = Wire(new ComponentItem { Name = "Part1-1", Transform = new double[16] });

            Assert.True(root.ContainsKey("parent"));
            Assert.Equal(JTokenType.Null, root["parent"]!.Type);
            Assert.True(root.ContainsKey("path"));
            Assert.Equal(string.Empty, (string?)root["path"]);
            Assert.True(root.ContainsKey("configuration"));
            Assert.Equal(string.Empty, (string?)root["configuration"]);
        }

        [Fact]
        public void InstanceWithoutRevisionOrVersionHasNullKeys()
        {
            var root = Wire(new InstanceInfo { InstanceId = "i1", Mode = "background", Owned = true, Pid = 4242 });

            Assert.True(root.ContainsKey("version"));
            Assert.Equal(JTokenType.Null, root["version"]!.Type);
            Assert.True(root.ContainsKey("revision"));
            Assert.Equal(JTokenType.Null, root["revision"]!.Type);
        }

        [Fact]
        public void OptionalMembersStayOmittedWhenNull()
        {
            var segment = Wire(new SegmentItem { Id = "0", Type = "line", Length = 0.05 });
            var mate = Wire(new MateItem { Name = "Coincident1", Type = "Coincident", Alignment = "Aligned" });
            var property = Wire(new PropertyItem { Name = "Mass", Type = "text", Value = "1 kg" });
            var mass = Wire(new MassPropertiesResult { Mass = 1.5, CenterOfMass = new double[3], PrincipalMoments = new double[3], MomentsOfInertia = new double[9] });
            var feature = Wire(new FeatureItem { Name = "Boss", Type = "Extrusion", Id = "7" });
            var dimension = Wire(new DimensionItem { Name = "D1", Value = 0.02, Type = "linear" });

            foreach (var key in new[] { "start", "end", "center", "radius" })
            {
                Assert.False(segment.ContainsKey(key));
            }

            Assert.False(mate.ContainsKey("value"));
            Assert.False(property.ContainsKey("resolved_value"));
            Assert.False(mass.ContainsKey("bounding_box"));
            Assert.False(feature.ContainsKey("parent_names"));
            Assert.False(dimension.ContainsKey("tolerance"));
        }

        [Fact]
        public void MateAndFeatureErrorFieldsStayAlwaysPresent()
        {
            var mate = Wire(new MateItem { Name = "M", Type = "Coincident", Alignment = "Aligned" });

            Assert.Equal("0", (string?)mate["error_code"]);
            Assert.False((bool)mate["warning"]!);
        }

        [Fact]
        public void HealthyMateHasNullErrorAndWhatsWrong()
        {
            var mate = Wire(new MateItem { Name = "M", Type = "Coincident", Alignment = "Aligned" });

            Assert.True(mate.ContainsKey("error"));
            Assert.Equal(JTokenType.Null, mate["error"]!.Type);
            Assert.True(mate.ContainsKey("whats_wrong"));
            Assert.Equal(JTokenType.Null, mate["whats_wrong"]!.Type);
        }

        [Fact]
        public void MateWithErrorSerializesTheErrorObject()
        {
            var mate = Wire(new MateItem { Name = "M", Type = "Coincident", Alignment = "Aligned", ErrorCode = "47", WhatsWrong = "Cannot solve." });

            Assert.Equal("47", (string?)mate["error"]!["code"]);
            Assert.Equal("swFeatureErrorMateIlldefined", (string?)mate["error"]!["name"]);
            Assert.Equal("Cannot solve.", (string?)mate["whats_wrong"]);
        }

        [Fact]
        public void MateEntityAlwaysHasResolvedAndNullableComponentPath()
        {
            var entity = Wire(new MateEntity { Component = "", EntityType = "unknown" });

            Assert.True(entity.ContainsKey("component_path"));
            Assert.Equal(JTokenType.Null, entity["component_path"]!.Type);
            Assert.False((bool)entity["resolved"]!);
        }

        [Fact]
        public void ComponentAlwaysHasFileExistsAndSuppression()
        {
            var component = Wire(new ComponentItem { Name = "Part1-1", Transform = new double[16] });

            Assert.False((bool)component["file_exists"]!);
            Assert.Equal("resolved", (string?)component["suppression"]);
        }

        [Fact]
        public void PartSummaryOmitsMissingComponentCount()
        {
            var root = Wire(new ModelSummary { Type = "part", SolidBodyCount = 1 });

            Assert.False(root.ContainsKey("missing_component_count"));
        }

        [Fact]
        public void AssemblySummaryKeepsMissingComponentCount()
        {
            var root = Wire(new ModelSummary { Type = "assembly", ComponentCount = 4, MateCount = 2, MissingComponentCount = 0 });

            Assert.Equal(0, (int)root["missing_component_count"]!);
        }

        [Fact]
        public void DocumentOpenResultAlwaysHasLoadWarningsAndMissingReferences()
        {
            var result = DocumentOpenResult.From(
                new DocumentItem { DocumentId = "doc-1", InstanceId = "i1", Title = "A", Path = @"C:\A.sldasm", Type = "assembly" },
                new SessionDocumentInfo("A", @"C:\A.sldasm", "assembly", null, true, false));
            var root = Wire(result);

            Assert.Equal(JTokenType.Array, root["load_warnings"]!.Type);
            Assert.Empty((JArray)root["load_warnings"]!);
            Assert.Equal(JTokenType.Array, root["missing_references"]!.Type);
            Assert.Empty((JArray)root["missing_references"]!);
        }

        [Fact]
        public void DocumentOpenResultKeepsAtMostFiftyMissingReferences()
        {
            var missing = Enumerable.Range(0, 70).Select(i => @"C:\Parts\P" + i + ".sldprt").ToList();
            var result = DocumentOpenResult.From(
                new DocumentItem { DocumentId = "doc-1", Title = "A", Path = @"C:\A.sldasm", Type = "assembly" },
                new SessionDocumentInfo("A", @"C:\A.sldasm", "assembly", null, false, false, missingReferences: missing, loadWarningBits: 0x100000 | 0x2));

            Assert.Equal(50, result.MissingReferences.Count);
            Assert.Equal(new[] { "read_only", "references_not_found" }, result.LoadWarnings);
        }

        [Fact]
        public void DocumentCopyFollowsDocumentCloseInTheCatalog()
        {
            var names = OperationCatalog.Create().Select(operation => operation.Name).ToList();

            Assert.Equal(names.IndexOf("document.close") + 1, names.IndexOf("document.copy"));
            Assert.Equal(ProtocolInfo.Access.Read, OperationCatalog.Create().Single(operation => operation.Name == "document.copy").Access);
        }

        [Fact]
        public void DocumentCopyResultKeepsEmptyWarningsAndFilesLists()
        {
            var root = Wire(new DocumentCopyResult { Folder = @"C:\Work\copies\x\", Path = @"C:\Work\copies\x\A.sldasm" });

            Assert.Equal(JTokenType.Array, root["warnings"]!.Type);
            Assert.Equal(JTokenType.Array, root["files"]!.Type);
            Assert.Equal(0, (int)root["count"]!);
            Assert.Equal(JTokenType.Boolean, root["files_truncated"]!.Type);
            Assert.False((bool)root["files_truncated"]!);
        }
    }

    /// <summary>The same shape checked on a processor response line (fake session: a part without material, a healthy feature).</summary>
    public class SpecShapeProcessorTests
    {
        [Fact]
        public async Task ResponseKeepsRequiredNullsForPartWithoutMaterialAndHealthyFeature()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var summary = (await h.CallAsync("model.summary", Harness.Args(("document", Harness.S("doc-1")))))["result"]!;
            var feature = (await h.CallAsync("feature.get", Harness.Args(("document", Harness.S("doc-1")), ("feature", Harness.S("Boss-Extrude1")))))["result"]!;
            var configurations = (await h.CallAsync("configurations.list", Harness.Args(("document", Harness.S("doc-1")))))["result"]!;
            var item = (JObject)configurations["items"]![0]!;

            Assert.Equal(JTokenType.Null, summary["material"]!.Type);
            Assert.Equal(JTokenType.Null, feature["whats_wrong"]!.Type);
            Assert.Equal(JTokenType.Null, item["parent"]!.Type);
            Assert.Equal(JTokenType.Null, item["description"]!.Type);
            Assert.Equal("Sketch1", (string?)feature["sketch"]);
        }

        [Fact]
        public async Task ResponseLineContainsTheKeysAsWritten()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var line = await h.SendRawAsync(TestData.Request("model.summary", Harness.Args(("document", Harness.S("doc-1")))));

            Assert.Contains("\"material\":null", line);
            Assert.Contains("\"active_configuration\":\"Default\"", line);
            Assert.DoesNotContain("component_count", line);
        }

        [Fact]
        public async Task FeatureGetLineKeepsWhatsWrongNull()
        {
            var h = new Harness();
            await h.CallAsync("document.open", Harness.Args(("path", Harness.S(@"C:\Models\Bracket.sldprt"))));

            var line = await h.SendRawAsync(TestData.Request("feature.get", Harness.Args(("document", Harness.S("doc-1")), ("feature", Harness.S("Boss-Extrude1")))));

            Assert.Contains("\"whats_wrong\":null", line);
            Assert.Contains("\"sketch\":\"Sketch1\"", line);
        }
    }
}
