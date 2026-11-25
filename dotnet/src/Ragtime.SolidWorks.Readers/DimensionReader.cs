using System;
using System.Collections.Generic;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Display dimensions of features. Values come from GetSystemValue3 for the current configuration (SI). A
    /// dimension appears once per response (by full name), and its feature is the owner named in its full name.
    /// Stateless.
    /// </summary>
    internal static class DimensionReader
    {
        /// <summary>Display dimensions of a feature in order. Includes a sketch's dimensions when the feature absorbs the sketch.</summary>
        public static List<SW.DisplayDimension> DisplayDimensions(SW.Feature feature) => ComWalk.Chain<SW.DisplayDimension>(
            "The display dimension list",
            ComWalk.DimensionLimit,
            () => feature.GetFirstDisplayDimension() as SW.DisplayDimension,
            display => feature.GetNextDisplayDimension(display) as SW.DisplayDimension);

        /// <summary>
        /// Dimension items of the features in order. A full name seen before in this response is skipped, so a
        /// dimension listed under two features (a sketch and the feature that absorbs it) appears once.
        /// </summary>
        public static List<DimensionItem> Describe(IEnumerable<SW.Feature> features)
        {
            var items = new List<DimensionItem>();
            var fullNames = new HashSet<string>(StringComparer.Ordinal);
            foreach (var feature in features)
            {
                foreach (var display in DisplayDimensions(feature))
                {
                    var dimension = display.GetDimension2(0);
                    if (dimension == null)
                    {
                        continue;
                    }

                    var fullName = dimension.FullName ?? string.Empty;
                    if (fullName.Length > 0 && !fullNames.Add(fullName))
                    {
                        continue;
                    }

                    items.Add(new DimensionItem
                    {
                        FullName = fullName,
                        Name = dimension.Name ?? string.Empty,
                        Feature = OwnerOf(fullName),
                        Value = Value(dimension),
                        Type = ReaderNames.DimensionType(display.GetType()),
                        Driven = dimension.DrivenState == (int)SWC.swDimensionDrivenState_e.swDimensionDriven,
                        ReadOnly = dimension.ReadOnly,
                        Tolerance = Tolerance(dimension),
                        Units = ProtocolInfo.SiUnits,
                    });
                }
            }

            return items;
        }

        /// <summary>First value of a feature's display dimensions in SI units, or null (used for mate values).</summary>
        public static double? FirstValue(SW.Feature feature)
        {
            foreach (var display in DisplayDimensions(feature))
            {
                var dimension = display.GetDimension2(0);
                if (dimension != null)
                {
                    return Value(dimension);
                }
            }

            return null;
        }

        /// <summary>
        /// The owning feature named in a full dimension name: "D1@Sketch1@bolt.Part" gives "Sketch1". Null when the name has
        /// no feature part.
        /// </summary>
        internal static string? OwnerOf(string fullName)
        {
            var parts = fullName.Split('@');
            return parts.Length >= 2 && parts[1].Length > 0 ? parts[1] : null;
        }

        /// <summary>The system value for the current configuration, falling back to SystemValue. Never zero-filled.</summary>
        private static double Value(SW.Dimension dimension) =>
            ComHelper.FirstDouble(dimension.GetSystemValue3((int)SWC.swInConfigurationOpts_e.swThisConfiguration, null))
            ?? dimension.SystemValue;

        /// <summary>
        /// Tolerance when the tolerance type is not none. Optional: null when the type or either limit cannot be read, so a
        /// missing limit is omitted rather than reported as zero.
        /// </summary>
        private static ToleranceInfo? Tolerance(SW.Dimension dimension)
        {
            var tolerance = ComHelper.TryObject(() => dimension.Tolerance);
            if (tolerance == null)
            {
                return null;
            }

            var type = ComHelper.TryValue(() => tolerance.Type);
            if (type == null || type == (int)SWC.swTolType_e.swTolNONE)
            {
                return null;
            }

            var max = ComHelper.TryValue(() => tolerance.GetMaxValue());
            var min = ComHelper.TryValue(() => tolerance.GetMinValue());
            if (max == null || min == null)
            {
                return null;
            }

            return new ToleranceInfo
            {
                Type = ReaderNames.ToleranceType(type.Value),
                Max = max.Value,
                Min = min.Value,
            };
        }
    }
}
