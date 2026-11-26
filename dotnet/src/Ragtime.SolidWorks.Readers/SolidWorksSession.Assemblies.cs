using System;
using System.Collections.Generic;
using System.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Assembly components and mates. Components come from AssemblyDoc.GetComponents; mates from the MateGroup feature.
    /// Assembly-only operations on other document types give not_supported_for_document_type.
    /// </summary>
    internal sealed partial class SolidWorksSession
    {
        /// <summary>Longest component chain followed; a deeper or cyclic chain fails.</summary>
        private const int MaxComponentDepth = 64;

        /// <summary>parentIndex: the component has no parent.</summary>
        private const int NoParent = -1;

        /// <summary>parentIndex: the parent is not in the listed components (top-level listing).</summary>
        private const int NotListed = -2;

        /// <summary>Entity type of a mate entity SolidWorks cannot return.</summary>
        private const string UnknownEntityType = "unknown";

        /// <summary>Mates of an assembly. Core rejects non-assemblies before calling.</summary>
        public IReadOnlyList<MateItem> ListMates(DocumentRef document) => Run("listing the mates", () =>
        {
            var model = RequireOpenDocument(document);
            RequireAssembly(model);
            var whatsWrong = WhatsWrongTable.Read(model);
            return MateFeatures(model).Select(pair => MateItemOf(pair.Feature, pair.Mate, whatsWrong)).ToList();
        });

        /// <summary>Components of an assembly; top level only when <paramref name="topLevelOnly"/> is true.</summary>
        public IReadOnlyList<ComponentItem> ListComponents(DocumentRef document, bool topLevelOnly) => Run("listing the components", () =>
        {
            var assembly = AssemblyOf(RequireOpenDocument(document));
            var components = ComWalk.Collection(assembly.GetComponents(topLevelOnly), "The component list")
                .OfType<SW.Component2>()
                .ToList();
            return ComponentItems(components, topLevelOnly);
        });

        private static void RequireAssembly(SW.ModelDoc2 model)
        {
            if (ReaderNames.DocumentType(model.GetType()) != ProtocolInfo.DocumentTypes.Assembly)
            {
                throw HostError.Coded(ErrorCodes.NotSupportedForDocumentType, "This operation needs an assembly document.");
            }
        }

        private static SW.AssemblyDoc AssemblyOf(SW.ModelDoc2 model)
        {
            RequireAssembly(model);
            return ComHelper.Require(model as SW.AssemblyDoc, "the assembly document");
        }

        /// <summary>Mate features under the MateGroup, with their Mate2 objects.</summary>
        private static List<(SW.Feature Feature, SW.Mate2 Mate)> MateFeatures(SW.ModelDoc2 model)
        {
            var pairs = new List<(SW.Feature Feature, SW.Mate2 Mate)>();
            foreach (var group in FeatureTree.TopLevel(model)
                .Where(feature => string.Equals(FeatureTree.TypeName(feature), "MateGroup", StringComparison.OrdinalIgnoreCase)))
            {
                foreach (var sub in FeatureTree.SubFeatures(group))
                {
                    var mate = sub.GetSpecificFeature2() as SW.Mate2;
                    if (mate != null)
                    {
                        pairs.Add((sub, mate));
                    }
                }
            }

            return pairs;
        }

        private static MateItem MateItemOf(SW.Feature feature, SW.Mate2 mate, WhatsWrongTable whatsWrong)
        {
            var health = FeatureTree.Health(feature);
            var type = ReaderNames.MateType(mate.Type);
            return new MateItem
            {
                Name = FeatureTree.Name(feature),
                Type = type,
                Alignment = ReaderNames.MateAlignment(mate.Alignment),
                Suppressed = FeatureTree.IsSuppressed(feature),
                ErrorCode = health.ErrorCode,
                Warning = health.Warning,
                WhatsWrong = health.HasError ? whatsWrong.For(FeatureTree.Name(feature)) : null,
                Value = type == "distance" || type == "angle" ? DimensionReader.FirstValue(feature) : null,
                Entities = MateEntities(mate),
            };
        }

        private static List<MateEntity> MateEntities(SW.Mate2 mate)
        {
            var entities = new List<MateEntity>();
            var count = mate.GetMateEntityCount();
            ComWalk.RequireWithinLimit(count, ComWalk.ItemLimit, "The mate entity list");
            for (var index = 0; index < count; index++)
            {
                entities.Add(MateEntityAt(mate, index));
            }

            return entities;
        }

        /// <summary>
        /// One mate entity. An entity SolidWorks cannot return (a null result or a read that throws) is listed as unknown
        /// and unresolved, so one broken reference does not fail the whole list.
        /// </summary>
        private static MateEntity MateEntityAt(SW.Mate2 mate, int index)
        {
            var entity = ComHelper.TryObject(() => mate.MateEntity(index));
            if (entity == null)
            {
                return new MateEntity
                {
                    Component = string.Empty,
                    ComponentPath = null,
                    EntityType = UnknownEntityType,
                    Resolved = false,
                };
            }

            var type = ComHelper.TryValue(() => entity.ReferenceType2);
            var component = ReferenceComponentOf(entity, out var readFailed);
            var componentName = ComHelper.TryText(() => component?.Name2);
            return new MateEntity
            {
                Component = componentName ?? string.Empty,
                ComponentPath = MateEntityRules.ComponentPath(componentName),
                EntityType = type.HasValue ? ReaderNames.SelectType(type.Value) : UnknownEntityType,
                Resolved = MateEntityRules.Resolved(
                    readFailed,
                    component != null,
                    component != null && ComponentFiles.IsResolved(component)),
            };
        }

        /// <summary>
        /// The reference component of an entity. <paramref name="readFailed"/> is true when the read threw, which is not the
        /// same as a legitimate null (no reference component, for example an assembly-level plane).
        /// </summary>
        private static SW.Component2? ReferenceComponentOf(SW.MateEntity2 entity, out bool readFailed)
        {
            try
            {
                readFailed = false;
                return entity.ReferenceComponent;
            }
            catch (Exception)
            {
                readFailed = true;
                return null;
            }
        }

        /// <summary>
        /// Items for the listed components. Each component's parent is read once; depth comes from the parent links
        /// inside the list, and child counts come from the same pass when all components are listed.
        /// </summary>
        private static List<ComponentItem> ComponentItems(List<SW.Component2> components, bool topLevelOnly)
        {
            var parents = components.Select(component => component.GetParent()).ToList();
            var indexOf = new Dictionary<IntPtr, int>();
            for (var index = 0; index < components.Count; index++)
            {
                indexOf[ComWalk.Identity(components[index])] = index;
            }

            var parentIndex = new int[components.Count];
            for (var index = 0; index < components.Count; index++)
            {
                var parent = parents[index];
                parentIndex[index] = parent == null
                    ? NoParent
                    : indexOf.TryGetValue(ComWalk.Identity(parent), out var found) ? found : NotListed;
            }

            var childCounts = new int[components.Count];
            if (!topLevelOnly)
            {
                foreach (var index in parentIndex.Where(parent => parent >= 0))
                {
                    childCounts[index]++;
                }
            }

            var items = new List<ComponentItem>(components.Count);
            for (var index = 0; index < components.Count; index++)
            {
                var children = topLevelOnly
                    ? ComWalk.Collection(components[index].GetChildren(), "The child component list").Count
                    : childCounts[index];
                items.Add(ComponentItemOf(components[index], DepthOf(index, parents, parentIndex), parents[index], children));
            }

            return items;
        }

        /// <summary>
        /// Number of ancestors of a listed component. Parents inside the list are followed without COM calls; a parent
        /// outside the list is walked up by COM.
        /// </summary>
        private static int DepthOf(int index, IReadOnlyList<SW.Component2> parents, IReadOnlyList<int> parentIndex)
        {
            var depth = 0;
            var current = index;
            while (parents[current] is SW.Component2 parent)
            {
                if (parentIndex[current] == NotListed)
                {
                    return depth + AncestorChainLength(parent);
                }

                depth++;
                if (depth > MaxComponentDepth)
                {
                    throw ComWalk.LimitReached("The component tree", MaxComponentDepth);
                }

                current = parentIndex[current];
            }

            return depth;
        }

        /// <summary>The component and its ancestors (COM walk), with the same depth cap as <see cref="DepthOf"/>.</summary>
        private static int AncestorChainLength(SW.Component2 start)
        {
            var count = 0;
            SW.Component2? current = start;
            while (current != null)
            {
                count++;
                if (count > MaxComponentDepth)
                {
                    throw ComWalk.LimitReached("The component tree", MaxComponentDepth);
                }

                current = current.GetParent();
            }

            return count;
        }

        private static ComponentItem ComponentItemOf(SW.Component2 component, int depth, SW.Component2? parent, int childrenCount)
        {
            var path = component.GetPathName() ?? string.Empty;
            var fileExists = ComponentFiles.LocalFileExists(path);
            return new ComponentItem
            {
                Name = component.Name2 ?? string.Empty,
                Path = path,
                Configuration = component.ReferencedConfiguration ?? string.Empty,
                Suppressed = component.IsSuppressed(),
                Hidden = component.Visible == (int)SWC.swComponentVisibilityState_e.swComponentHidden,
                Fixed = component.IsFixed(),
                IsVirtual = component.IsVirtual,
                Depth = depth,
                Parent = ComHelper.TryText(() => parent?.Name2),
                Transform = TransformOf(component),
                ChildrenCount = childrenCount,
                FileExists = fileExists,
                Suppression = fileExists
                    ? ComponentFiles.SuppressionOf(component.GetSuppression2())
                    : ComponentSuppression.MissingFile,
            };
        }

        /// <summary>
        /// The component transform as the 16 raw values of MathTransform.ArrayData, in SolidWorks' own order (not
        /// reordered). Required: throws solidworks_error when missing.
        /// </summary>
        private static double[] TransformOf(SW.Component2 component)
        {
            var transform = ComHelper.Require(component.Transform, "the component transform");
            return ComHelper.RequireDoubles(transform.ArrayData, 16, "the component transform values");
        }
    }
}
