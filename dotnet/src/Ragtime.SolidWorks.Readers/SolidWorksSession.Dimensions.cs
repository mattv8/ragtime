using System.Collections.Generic;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>Dimensions of one feature or of every top-level feature.</summary>
    internal sealed partial class SolidWorksSession
    {
        /// <summary>
        /// All dimensions, once each, or the dimensions of <paramref name="feature"/> when it is not null (including a
        /// sketch's dimensions when that feature absorbs the sketch; their feature is then the sketch).
        /// </summary>
        public IReadOnlyList<DimensionItem> ListDimensions(DocumentRef document, string? feature) => Run("listing the dimensions", () =>
        {
            var model = RequireOpenDocument(document);
            if (feature != null)
            {
                var found = FeatureTree.Find(model, feature) ?? throw ComHelper.NotFound("feature", feature);
                return DimensionReader.Describe(new[] { found });
            }

            return DimensionReader.Describe(FeatureTree.TopLevel(model));
        });
    }
}
