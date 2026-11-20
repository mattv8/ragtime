using System;
using System.Collections.Generic;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>Custom properties (file or configuration), mass properties and equations.</summary>
    internal sealed partial class SolidWorksSession
    {
        /// <summary>
        /// File-level properties, or the properties of <paramref name="configuration"/> when it is not null. Values are read
        /// from SolidWorks' cache without evaluating or activating any configuration.
        /// </summary>
        public PropertiesResult GetProperties(DocumentRef document, string? configuration) => Run("reading the custom properties", () =>
        {
            var model = RequireOpenDocument(document);
            var name = configuration ?? string.Empty;
            if (name.Length > 0 && !ComHelper.Strings(model.GetConfigurationNames()).Contains(name))
            {
                throw ComHelper.NotFound("configuration", name);
            }

            var manager = model.Extension.CustomPropertyManager[name];
            var names = ComHelper.Strings(manager.GetNames());
            ComWalk.RequireWithinLimit(names.Count, ComWalk.ItemLimit, "The custom property list");
            var result = new PropertiesResult
            {
                Scope = name.Length == 0 ? "file" : "configuration:" + name,
            };
            foreach (var property in names)
            {
                result.Properties.Add(PropertyItemOf(manager, property));
            }

            return result;
        });

        /// <summary>Mass properties in SI units. Throws solidworks_error when the model has no solid body or a value is missing.</summary>
        public MassPropertiesResult GetMassProperties(DocumentRef document) => Run("reading the mass properties", () =>
        {
            var model = RequireOpenDocument(document);
            if (ReaderNames.DocumentType(model.GetType()) == ProtocolInfo.DocumentTypes.Drawing)
            {
                throw HostError.Coded(ErrorCodes.NotSupportedForDocumentType, "Mass properties need a part or an assembly.");
            }

            var mass = ComHelper.Require(model.Extension.CreateMassProperty(), "mass properties (the model may have no solid body)");
            return new MassPropertiesResult
            {
                Mass = mass.Mass,
                Volume = mass.Volume,
                SurfaceArea = mass.SurfaceArea,
                Density = mass.Density,
                CenterOfMass = ComHelper.RequireDoubles(mass.CenterOfMass, 3, "the centre of mass"),
                PrincipalMoments = ComHelper.RequireDoubles(mass.PrincipleMomentsOfInertia, 3, "the principal moments of inertia"),
                MomentsOfInertia = ComHelper.RequireDoubles(
                    mass.GetMomentOfInertia((int)SWC.swMassPropertyMoment_e.swMassPropertyMomentAboutCenterOfMass),
                    9,
                    "the moments of inertia"),
                BoundingBox = BoundingBox(model),
                Units = ProtocolInfo.SiUnits,
            };
        });

        /// <summary>All equations of the model, by index. A model without an equation manager has none.</summary>
        public IReadOnlyList<EquationItem> ListEquations(DocumentRef document) => Run("listing the equations", () =>
        {
            var model = RequireOpenDocument(document);
            var items = new List<EquationItem>();
            var manager = model.GetEquationMgr();
            if (manager == null)
            {
                return items;
            }

            var count = manager.GetCount();
            ComWalk.RequireWithinLimit(count, ComWalk.ItemLimit, "The equation list");
            for (var index = 0; index < count; index++)
            {
                items.Add(new EquationItem
                {
                    Index = index,
                    Equation = manager.Equation[index] ?? string.Empty,
                    Value = FiniteOrNull(ComHelper.TryValue(() => manager.Value[index])),
                    IsGlobalVariable = manager.GlobalVariable[index],
                    Suppressed = manager.Suppression[index],
                });
            }

            return items;
        });

        /// <summary>
        /// One property. Get5 with useCached = true returns the stored value and, only when SolidWorks reports it as
        /// resolved, the resolved value; otherwise the resolved value is empty.
        /// </summary>
        private static PropertyItem PropertyItemOf(SW.CustomPropertyManager manager, string name)
        {
            manager.Get5(name, true, out string value, out string resolved, out bool wasResolved);
            return new PropertyItem
            {
                Name = name,
                Type = ReaderNames.CustomInfoType(manager.GetType2(name)),
                Value = value ?? string.Empty,
                ResolvedValue = wasResolved ? resolved : null,
            };
        }

        /// <summary>
        /// Bounding box of a part, or null when it cannot be read (optional field). Never zero-filled.
        /// </summary>
        private static double[]? BoundingBox(SW.ModelDoc2 model)
        {
            var part = model as SW.PartDoc;
            if (part == null)
            {
                return null;
            }

            var box = ComHelper.Doubles(ComHelper.TryObject(() => part.GetPartBox(true)));
            return box.Length == 6 ? box : null;
        }

        private static double? FiniteOrNull(double? value) =>
            value.HasValue && !double.IsNaN(value.Value) && !double.IsInfinity(value.Value) ? value : null;
    }
}
