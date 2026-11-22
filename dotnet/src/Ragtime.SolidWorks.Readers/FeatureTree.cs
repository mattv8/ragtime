using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Feature tree walks and per-feature facts. Top-level walks use FirstFeature and GetNextFeature; sub-features
    /// are read only when a caller asks for them. Walks are cycle-guarded and capped (see <see cref="ComWalk"/>).
    /// Stateless.
    /// </summary>
    internal static class FeatureTree
    {
        private static readonly string[] SystemFolderTypes =
        {
            "CommentsFolder", "FavoriteFolder", "HistoryFolder", "SelectionSetFolder", "SensorFolder", "DocsFolder",
            "DetailCabinet", "InkMarkupFolder", "SurfaceBodyFolder", "SolidBodyFolder", "EnvFolder", "EqnFolder",
            "MaterialFolder", "ConfigTableFolder",
        };

        /// <summary>Top-level features in tree order.</summary>
        public static List<SW.Feature> TopLevel(SW.ModelDoc2 model) => ComWalk.Chain<SW.Feature>(
            "The feature tree",
            ComWalk.FeatureLimit,
            () => model.FirstFeature() as SW.Feature,
            feature => feature.GetNextFeature() as SW.Feature);

        /// <summary>Top-level features without system folders (Comments, Sensors and similar).</summary>
        public static List<SW.Feature> Visible(SW.ModelDoc2 model) =>
            TopLevel(model).Where(feature => !IsSystemFolder(TypeName(feature))).ToList();

        /// <summary>Direct sub-features of a feature in order.</summary>
        public static List<SW.Feature> SubFeatures(SW.Feature feature) => ComWalk.Chain<SW.Feature>(
            "The sub-feature list",
            ComWalk.FeatureLimit,
            () => feature.GetFirstSubFeature() as SW.Feature,
            current => current.GetNextSubFeature() as SW.Feature);

        /// <summary>The feature with that name, found by FeatureByName on a part or assembly, else in the top-level tree.</summary>
        public static SW.Feature? Find(SW.ModelDoc2 model, string name) =>
            ByName(model, name) ?? TopLevel(model).FirstOrDefault(feature => string.Equals(Name(feature), name, StringComparison.OrdinalIgnoreCase));

        public static string Name(SW.Feature feature) => feature.Name ?? string.Empty;

        public static string TypeName(SW.Feature feature) => feature.GetTypeName2() ?? string.Empty;

        public static string Id(SW.Feature feature) => ComHelper.IntText(feature.GetID());

        public static bool IsSystemFolder(string typeName) =>
            Array.Exists(SystemFolderTypes, folder => string.Equals(folder, typeName, StringComparison.OrdinalIgnoreCase));

        /// <summary>True for a sketch: a 2D profile feature or a 3D sketch.</summary>
        public static bool IsSketch(SW.Feature feature)
        {
            var typeName = TypeName(feature);
            return string.Equals(typeName, "ProfileFeature", StringComparison.OrdinalIgnoreCase)
                || string.Equals(typeName, "3DProfileFeature", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Suppression in the active configuration (IsSuppressed2), falling back to IsSuppressed.</summary>
        public static bool IsSuppressed(SW.Feature feature)
        {
            var state = feature.IsSuppressed2((int)SWC.swInConfigurationOpts_e.swThisConfiguration, null);
            var first = state is Array ? ComHelper.Objects(state).FirstOrDefault() : state;
            return first == null
                ? feature.IsSuppressed()
                : Convert.ToBoolean(first, CultureInfo.InvariantCulture);
        }

        /// <summary>Error or warning state from GetErrorCode2.</summary>
        public static FeatureHealth Health(SW.Feature feature)
        {
            var code = feature.GetErrorCode2(out bool warning);
            return new FeatureHealth(code, warning);
        }

        /// <summary>Names of features, skipping anything that is not a feature.</summary>
        public static List<string> Names(object? features) =>
            ComHelper.Objects(features).OfType<SW.Feature>().Select(Name).ToList();

        /// <summary>The parent features of a feature (the sketch and reference planes it is built from). Empty when unavailable.</summary>
        public static List<SW.Feature> Parents(SW.Feature feature) =>
            ComHelper.Objects(ComHelper.TryObject(() => feature.GetParents())).OfType<SW.Feature>().ToList();

        /// <summary>Feature list item for features.list.</summary>
        public static FeatureItem Item(SW.Feature feature)
        {
            var health = Health(feature);
            return new FeatureItem
            {
                Name = Name(feature),
                Type = TypeName(feature),
                Id = Id(feature),
                Suppressed = IsSuppressed(feature),
                ErrorCode = health.ErrorCode,
                Warning = health.Warning,
                DimensionCount = DimensionReader.DisplayDimensions(feature).Count,
                ParentNames = Names(ComHelper.TryObject(() => feature.GetParents())),
            };
        }

        /// <summary>Full feature detail for feature.get.</summary>
        public static FeatureDetail Detail(SW.ModelDoc2 model, SW.Feature feature)
        {
            var health = Health(feature);
            return new FeatureDetail
            {
                Name = Name(feature),
                Type = TypeName(feature),
                Id = Id(feature),
                Suppressed = IsSuppressed(feature),
                ErrorCode = health.ErrorCode,
                Warning = health.Warning,
                WhatsWrong = health.HasError ? WhatsWrong(model, feature) : null,
                Parents = Names(ComHelper.TryObject(() => feature.GetParents())),
                Children = Names(feature.GetChildren()),
                Dimensions = DimensionReader.Describe(new[] { feature }),
                Sketch = SketchName(feature),
            };
        }

        /// <summary>
        /// Text explaining the feature's error, from ModelDocExtension.GetWhatsWrong. The third array is taken as the
        /// messages, aligned with the feature names (not verified live). Optional: empty when unavailable.
        /// </summary>
        private static string WhatsWrong(SW.ModelDoc2 model, SW.Feature feature)
        {
            try
            {
                model.Extension.GetWhatsWrong(out object names, out object codes, out object messages);
                var index = ComHelper.Strings(names).IndexOf(Name(feature));
                var texts = ComHelper.Strings(messages);
                return index >= 0 && index < texts.Count ? texts[index] : string.Empty;
            }
            catch (Exception)
            {
                return string.Empty;
            }
        }

        /// <summary>The sketch a feature is built from: its own sub-feature or parent of sketch type.</summary>
        private static string? SketchName(SW.Feature feature)
        {
            var sketch = SubFeatures(feature).FirstOrDefault(IsSketch) ?? Parents(feature).FirstOrDefault(IsSketch);
            return sketch == null ? null : Name(sketch);
        }

        private static SW.Feature? ByName(SW.ModelDoc2 model, string name)
        {
            var part = model as SW.PartDoc;
            if (part != null)
            {
                return ComHelper.TryObject(() => part.FeatureByName(name) as SW.Feature);
            }

            var assembly = model as SW.AssemblyDoc;
            if (assembly != null)
            {
                return ComHelper.TryObject(() => assembly.FeatureByName(name) as SW.Feature);
            }

            return null;
        }
    }
}
