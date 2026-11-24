using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// The v1 operation table, including the PDM read and write operations.
    /// host.shutdown is internal (used by <c>stop</c>) and never listed by describe.
    /// </summary>
    public static class OperationCatalog
    {
        private const string DocumentArgument = "Document id from documents.list or document.open (doc-N), or the full path of an open document.";
        private const string SketchPartDescription = "Which part of the sketch: summary (default, not paged), segments, points, relations or dimensions.";

        /// <summary>Builds the definitions in describe order.</summary>
        public static List<OperationDefinition> Create() => new List<OperationDefinition>
        {
            new OperationDefinition(
                "describe",
                "Lists every operation with its summary, access, paging, argument schema and result.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: SchemaTypes.NoArguments(),
                result: "Protocol, host_version and the operations array (name, summary, access, paged, arguments, result)."),

            new OperationDefinition(
                "status",
                "Reports the Windows account the SolidWorks agent runs as, its desktop session, installed SolidWorks versions and the instances.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: SchemaTypes.NoArguments(),
                result: "host_version, protocol, account {name, sid, is_administrator}, session {id, interactive}, solidworks {installed_versions, instances (version and revision are null when unknown)}, uptime_seconds."),

            new OperationDefinition(
                "documents.list",
                "Lists the documents the SolidWorks agent knows about in every instance.",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema().Paged().Build(),
                result: "Paged items: document_id, instance_id, title, path, type, active_configuration, opened_by_host, read_only, modified. path is empty for an unsaved document; active_configuration is null when none. read_only reflects how SolidWorks opened the document. Referenced component documents appear with opened_by_host false."),

            new OperationDefinition(
                "document.open",
                "Opens a local SolidWorks file read-only and silently in a background instance (not in attach mode).",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: new ObjectSchema()
                    .Property("path", SchemaTypes.Text("Absolute local drive path ending in .sldprt, .sldasm or .slddrw, such as C:\\Models\\part.sldprt. UNC, \\\\?\\ and relative paths are refused."), required: true)
                    .Property("configuration", SchemaTypes.Text("Configuration to activate. Default: the file's active configuration."))
                    .Build(),
                result: "The document item (see documents.list)."),

            new OperationDefinition(
                "document.close",
                "Closes a document that the SolidWorks agent opened. Never saves.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Build(),
                result: "{closed: true}."),

            new OperationDefinition(
                "model.summary",
                "Summarises one document: type, counts of features, errors and warnings, material, units.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Build(),
                result: "document_id, type, title, path, active_configuration, configuration_count, feature_count, features_with_errors, features_with_warnings, material, custom_property_count, needs_rebuild, units, plus solid_body_count (parts) or component_count and mate_count (assemblies). active_configuration and material are always present, null when there is no value (material is null for a part with no material assigned)."),

            new OperationDefinition(
                "configurations.list",
                "Lists the configurations of a document.",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Paged()
                    .Build(),
                result: "Paged items: name, is_active, description, parent. description and parent are always present, null when none; parent is null for a root configuration."),

            new OperationDefinition(
                "features.list",
                "Lists the feature tree in tree order. System folders are hidden unless include_system is true.",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Property("include_system", SchemaTypes.Boolean("Include system folders such as Comments and Sensors. Default false."))
                    .Property("types", SchemaTypes.StringArray("Only features of these type names, e.g. [\"Extrusion\", \"Sketch\"].", 50))
                    .Paged()
                    .Build(),
                result: "Paged items: name, type, id, suppressed, error_code, warning, dimension_count, parent_names (optional)."),

            new OperationDefinition(
                "feature.get",
                "Returns one feature with its parents, children, dimensions and sketch.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Property("feature", SchemaTypes.Text("Feature name as shown in the tree."), required: true)
                    .Build(),
                result: "name, type, id, suppressed, error_code, warning, whats_wrong, parents, children, dimensions (dimension items), sketch (optional). whats_wrong is always present, null when the feature is healthy."),

            new OperationDefinition(
                "sketch.get",
                "Returns a sketch: summary by default, or its segments, points, relations or dimensions (paged).",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Property("sketch", SchemaTypes.Text("Sketch name as shown in the tree."), required: true)
                    .Property("part", SchemaTypes.Enum(SketchPartDescription, "summary", "segments", "points", "relations", "dimensions"))
                    .Paged()
                    .Build(),
                result: "For part summary: name, is_3d, constrained_status, segment_count, point_count, relation_count, dimension_count, plane_or_face (always present, null when not known), units (not paged). Other parts are pages of segment, point, relation or dimension items in SI units."),

            new OperationDefinition(
                "dimensions.list",
                "Lists dimensions of the document, or only those of one feature.",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Property("feature", SchemaTypes.Text("Feature name; omit for all dimensions."))
                    .Paged()
                    .Build(),
                result: "Paged items: full_name, name, feature (null when no feature owns the dimension), value, type, driven, read_only, tolerance (optional), units."),

            new OperationDefinition(
                "mates.list",
                "Lists the mates of an assembly. Not supported for parts or drawings.",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Paged()
                    .Build(),
                result: "Paged items: name, type, alignment, suppressed, error_code, warning, value (optional), entities [{component, entity_type}]."),

            new OperationDefinition(
                "components.list",
                "Lists the components of an assembly. Not supported for parts or drawings.",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Property("top_level_only", SchemaTypes.Boolean("Only top-level components. Default true."))
                    .Paged()
                    .Build(),
                result: "Paged items: name, path, configuration, suppressed, hidden, fixed, is_virtual, depth, parent, transform, children_count. parent is null for a top-level component. transform is 16 numbers in SolidWorks MathTransform.ArrayData order: [0-8] rotation matrix row by row, [9-11] translation in metres, [12] scale, [13-15] unused."),

            new OperationDefinition(
                "properties.get",
                "Returns the custom properties of the file, or of one configuration.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Property("configuration", SchemaTypes.Text("Configuration name. Omit for file-level properties."))
                    .Build(),
                result: "{scope, properties: [{name, type, value, resolved_value}]}. resolved_value is present only when SolidWorks has an evaluated value cached; the tool never forces evaluation."),

            new OperationDefinition(
                "mass_properties.get",
                "Returns mass, volume, surface area, centre of mass and inertia of a part or assembly.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Build(),
                result: "mass (kg), volume (m3), surface_area (m2), density (kg/m3), center_of_mass [x,y,z] (m), principal_moments [3], moments_of_inertia [9] (kg m2), bounding_box [6] (optional), units."),

            new OperationDefinition(
                "equations.list",
                "Lists the equations of a document.",
                ProtocolInfo.Access.Read,
                paged: true,
                argumentsSchema: new ObjectSchema()
                    .Property("document", SchemaTypes.Document(DocumentArgument), required: true)
                    .Paged()
                    .Build(),
                result: "Paged items: index, equation, value (null when SolidWorks has no finite value), is_global_variable, suppressed."),

            new OperationDefinition(
                "pdm.status", "Reports PDM client and linked vault status.", ProtocolInfo.Access.Read, false,
                SchemaTypes.NoArguments(),
                "client_installed, client_version, work_folder (string or null), views [{vault, root, logged_in, user}], linked {vault, root, logged_in, user, writes_enabled, approved_folders, problem} or null."),
            new OperationDefinition(
                "pdm.folder.list", "Lists the direct children of a vault folder.", ProtocolInfo.Access.Read, true,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault folder path."), true).Paged().Build(),
                "Paged items: name, kind, vault_path, latest_version, local_version, checked_out."),
            new OperationDefinition(
                "pdm.file.info", "Reports live vault file state.", ProtocolInfo.Access.Read, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault file path."), true).Build(),
                "vault_path, local_path, name, file_id, exists_locally, local_version, latest_version, checked_out, checked_out_by, checked_out_on, checked_out_here, checked_out_by_tool, workflow_state, in_approved_folder."),
            new OperationDefinition(
                "pdm.variables.get", "Reads the data-card variables of a vault file.", ProtocolInfo.Access.Read, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault file path."), true)
                    .Property("configuration", SchemaTypes.Text("Configuration name; default @.")).Build(),
                "vault_path, configurations, configuration, variables [{name, value}]."),
            new OperationDefinition(
                "pdm.folder.rights", "Checks folder rights for the logged-in user.", ProtocolInfo.Access.Read, false,
                new ObjectSchema().Property("paths", new JObject
                {
                    ["type"] = "array",
                    ["maxItems"] = 25,
                    ["items"] = new JObject { ["type"] = "string", ["minLength"] = 1, ["maxLength"] = 260 },
                }, true).Build(),
                "user, items [{vault_path, checked, exists, rights {read, add_files, check_out, add_folders}}]."),
            new OperationDefinition(
                "pdm.folder.children", "Lists direct subfolders with folder rights.", ProtocolInfo.Access.Read, true,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault folder path."), true)
                    .Property("offset", SchemaTypes.Integer(0, int.MaxValue, "Index of the first item; default 0."))
                    .Property("limit", SchemaTypes.Integer(1, 100, "Maximum items; default 50."))
                    .Property("snapshot", SchemaTypes.Snapshot()).Build(),
                "Paged items: name, vault_path, rights {read, add_files, check_out, add_folders} or null."),
            new OperationDefinition(
                "pdm.file.get_latest", "Fetches the latest vault file and optionally its references.", ProtocolInfo.Access.Read, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault file path."), true)
                    .Property("include_references", SchemaTypes.Boolean("Include references; default true.")).Build(),
                "fetched [vault_path], skipped [{vault_path, reason}], file (file info)."),
            new OperationDefinition(
                "pdm.folder.create", "Creates a folder inside an approved vault folder.", ProtocolInfo.Access.Write, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault folder path."), true).Build(),
                "vault_path, local_path, created."),
            new OperationDefinition(
                "pdm.file.add", "Adds a file inside an approved vault folder.", ProtocolInfo.Access.Write, false,
                new ObjectSchema().Property("source", SchemaTypes.Text("Absolute local source file path."), true)
                    .Property("folder", SchemaTypes.Text("Existing vault folder path."), true)
                    .Property("name", SchemaTypes.Text("Target file name; defaults to source file name.")).Build(),
                "File info: vault_path, local_path, name, file_id, exists_locally, local_version, latest_version, checked_out, checked_out_by, checked_out_on, checked_out_here, checked_out_by_tool, workflow_state, in_approved_folder."),
            new OperationDefinition(
                "pdm.file.checkout", "Checks out a file inside an approved vault folder.", ProtocolInfo.Access.Write, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault file path."), true).Build(),
                "File info: vault_path, local_path, name, file_id, exists_locally, local_version, latest_version, checked_out, checked_out_by, checked_out_on, checked_out_here, checked_out_by_tool, workflow_state, in_approved_folder."),
            new OperationDefinition(
                "pdm.file.undo_checkout", "Undoes checkout of a file inside an approved vault folder.", ProtocolInfo.Access.Write, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault file path."), true).Build(),
                "File info: vault_path, local_path, name, file_id, exists_locally, local_version, latest_version, checked_out, checked_out_by, checked_out_on, checked_out_here, checked_out_by_tool, workflow_state, in_approved_folder."),
            new OperationDefinition(
                "pdm.file.checkin", "Checks in a file inside an approved vault folder.", ProtocolInfo.Access.Write, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault file path."), true)
                    .Property("comment", new JObject { ["type"] = "string", ["minLength"] = 1, ["maxLength"] = 500 }, true)
                    .Property("keep_checked_out", SchemaTypes.Boolean("Keep checked out; default false.")).Build(),
                "File info: vault_path, local_path, name, file_id, exists_locally, local_version, latest_version, checked_out, checked_out_by, checked_out_on, checked_out_here, checked_out_by_tool, workflow_state, in_approved_folder."),
            new OperationDefinition(
                "pdm.variables.set", "Sets data-card variables for a file inside an approved vault folder.", ProtocolInfo.Access.Write, false,
                new ObjectSchema().Property("path", SchemaTypes.Text("Vault file path."), true)
                    .Property("variables", new JObject {
                        ["type"] = "object", ["minProperties"] = 1, ["maxProperties"] = 50,
                        ["additionalProperties"] = new JObject { ["type"] = new JArray("string", "number", "boolean") },
                    }, true)
                    .Property("configuration", SchemaTypes.Text("Configuration name; default @.")).Build(),
                "vault_path, configuration, updated [name]."),

            new OperationDefinition(
                "script.run", "Runs a C# 5 script against a SolidWorks instance.", ProtocolInfo.Access.Script, false,
                new ObjectSchema()
                    .Property("source", new JObject
                    {
                        ["type"] = "string", ["minLength"] = 1, ["maxLength"] = 16384,
                        ["maxUtf8Bytes"] = 16384,
                        ["description"] = "C# 5 source, 1–16384 UTF-8 bytes.",
                    }, true)
                    .Property("purpose", new JObject
                    {
                        ["type"] = "string", ["minLength"] = 1, ["maxLength"] = 500,
                        ["pattern"] = @"^[^\x00-\x1F\x7F]+$(?![\s\S])",
                    }, true)
                    .Property("arguments", new JObject
                    {
                        ["type"] = "object", ["additionalProperties"] = true, ["maxJsonBytes"] = 4096,
                        ["description"] = "Script arguments; default {}; at most 4096 compact JSON UTF-8 bytes.",
                    })
                    .Property("instance", new JObject
                    {
                        ["type"] = "string", ["pattern"] = @"^(i[1-9][0-9]*|a[1-9][0-9]*)$(?![\s\S])",
                        ["description"] = "Optional instance id; defaults to the instance selected for document.open.",
                    }).Build(),
                "result, log, log_truncated, documents, instance_id, compile_ms, run_ms. Request timeout: at most 600 seconds."),

            new OperationDefinition(
                "host.shutdown",
                "Internal: asks the SolidWorks agent to close its owned instances and exit.",
                ProtocolInfo.Access.Read,
                paged: false,
                argumentsSchema: new ObjectSchema()
                    .Property("if_idle_seconds", SchemaTypes.Integer(0, 86400, "Refuse with host_busy unless the SolidWorks agent has been idle this many seconds and nothing would be lost."))
                    .Build(),
                result: "{stopping: true}, or host_busy with details.reasons when if_idle_seconds is given and the SolidWorks agent is in use.",
                isInternal: true),
        };
    }
}
