using System;
using System.Collections.Generic;
using System.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>Feature tree list and single-feature detail.</summary>
    internal sealed partial class SolidWorksSession
    {
        /// <summary>Top-level features in tree order, optionally without system folders and filtered by type name.</summary>
        public IReadOnlyList<FeatureItem> ListFeatures(DocumentRef document, bool includeSystem, IReadOnlyList<string>? types) =>
            Run("listing the features", () =>
            {
                var model = RequireOpenDocument(document);
                var items = new List<FeatureItem>();
                foreach (var feature in FeatureTree.TopLevel(model))
                {
                    var typeName = FeatureTree.TypeName(feature);
                    if (!includeSystem && FeatureTree.IsSystemFolder(typeName))
                    {
                        continue;
                    }

                    if (types != null && !types.Any(type => string.Equals(type, typeName, StringComparison.OrdinalIgnoreCase)))
                    {
                        continue;
                    }

                    items.Add(FeatureTree.Item(feature));
                }

                return items;
            });

        /// <summary>One feature with its parents, children, dimensions and sketch. Throws entity_not_found for an unknown name.</summary>
        public FeatureDetail GetFeature(DocumentRef document, string feature) => Run("reading the feature", () =>
        {
            var model = RequireOpenDocument(document);
            var found = FeatureTree.Find(model, feature) ?? throw ComHelper.NotFound("feature", feature);
            return FeatureTree.Detail(model, found);
        });
    }
}
