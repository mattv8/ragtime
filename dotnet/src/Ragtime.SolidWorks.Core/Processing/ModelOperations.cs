using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Maps model operations to session calls and applies paging or the result budget. Runs on the instance's
    /// STA thread (inside the instance manager's work delegate).
    /// </summary>
    internal static class ModelOperations
    {
        /// <summary>Operations that only apply to assemblies.</summary>
        public static bool RequiresAssembly(string operation) =>
            operation == "mates.list" || operation == "components.list";

        /// <summary>Executes one model operation against the session.</summary>
        public static JToken Execute(
            string operation,
            ISolidWorksSession session,
            DocumentEntry document,
            JObject arguments,
            RequestContext context)
        {
            var reference = document.ToRef();
            switch (operation)
            {
                case "model.summary":
                {
                    var summary = session.GetModelSummary(reference);
                    summary.DocumentId = document.DocumentId;
                    return Fits(summary, context);
                }

                case "configurations.list":
                    return Page(session.ListConfigurations(reference), arguments, context);

                case "features.list":
                    return Page(
                        session.ListFeatures(
                            reference,
                            ArgumentReader.OptionalBoolean(arguments, "include_system", false),
                            ArgumentReader.OptionalStringList(arguments, "types")),
                        arguments,
                        context);

                case "feature.get":
                    return Fits(session.GetFeature(reference, ArgumentReader.RequiredString(arguments, "feature")), context);

                case "sketch.get":
                    return ExecuteSketch(session, reference, arguments, context);

                case "dimensions.list":
                    return Page(session.ListDimensions(reference, ArgumentReader.OptionalString(arguments, "feature")), arguments, context);

                case "mates.list":
                    return Page(session.ListMates(reference), arguments, context);

                case "components.list":
                    return Page(
                        session.ListComponents(reference, ArgumentReader.OptionalBoolean(arguments, "top_level_only", true)),
                        arguments,
                        context);

                case "properties.get":
                    return Fits(session.GetProperties(reference, ArgumentReader.OptionalString(arguments, "configuration")), context);

                case "mass_properties.get":
                    return Fits(session.GetMassProperties(reference), context);

                case "equations.list":
                    return Page(session.ListEquations(reference), arguments, context);

                default:
                    throw new InvalidOperationException($"No handler is registered for operation '{operation}'.");
            }
        }

        private static JToken ExecuteSketch(
            ISolidWorksSession session,
            DocumentRef reference,
            JObject arguments,
            RequestContext context)
        {
            var sketch = ArgumentReader.RequiredString(arguments, "sketch");
            var part = ArgumentReader.OptionalString(arguments, "part") ?? "summary";
            switch (part)
            {
                case "summary":
                    return Fits(session.GetSketchSummary(reference, sketch), context);
                case "segments":
                    return Page(session.ListSketchSegments(reference, sketch), arguments, context);
                case "points":
                    return Page(session.ListSketchPoints(reference, sketch), arguments, context);
                case "relations":
                    return Page(session.ListSketchRelations(reference, sketch), arguments, context);
                case "dimensions":
                    return Page(session.ListSketchDimensions(reference, sketch), arguments, context);
                default:
                    throw HostError.InvalidArguments("part", "'part' must be one of summary, segments, points, relations, dimensions.");
            }
        }

        private static JToken Page<T>(IReadOnlyList<T> items, JObject arguments, RequestContext context) =>
            Pager.Page(items, PageRequest.FromArguments(arguments), ProtocolInfo.SiUnits, context.MaxResultBytes);

        private static JToken Fits(object dto, RequestContext context) =>
            Pager.RequireFits(WireJson.ToToken(dto), context.MaxResultBytes);
    }
}
