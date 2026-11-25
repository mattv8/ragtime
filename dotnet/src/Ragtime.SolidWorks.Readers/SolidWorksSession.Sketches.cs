using System;
using System.Collections.Generic;
using System.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Sketch summary, segments, points and relations. Coordinates are sketch-space metres from the sketch point
    /// objects. Geometry points and radii are optional and nullable; counts, lengths and point positions are required.
    /// </summary>
    internal sealed partial class SolidWorksSession
    {
        /// <summary>Counts and constraint status of a sketch. Throws entity_not_found for an unknown sketch name.</summary>
        public SketchSummary GetSketchSummary(DocumentRef document, string sketch) => Run("reading the sketch summary", () =>
        {
            var model = RequireOpenDocument(document);
            var feature = SketchFeatureOf(model, sketch);
            var sketchObject = SketchObject(feature);
            return new SketchSummary
            {
                Name = FeatureTree.Name(feature),
                Is3D = sketchObject.Is3D(),
                ConstrainedStatus = ReaderNames.ConstrainedStatus(sketchObject.GetConstrainedStatus()),
                SegmentCount = Segments(sketchObject).Count,
                PointCount = Points(sketchObject).Count,
                RelationCount = Relations(sketchObject).Count,
                DimensionCount = DimensionReader.DisplayDimensions(feature).Count,
                PlaneOrFace = PlaneOrFace(feature),
                Units = ProtocolInfo.SiUnits,
            };
        });

        /// <summary>Segments of a sketch in sketch coordinates (metres).</summary>
        public IReadOnlyList<SegmentItem> ListSketchSegments(DocumentRef document, string sketch) => Run("listing the sketch segments", () =>
        {
            var sketchObject = SketchObject(SketchFeatureOf(RequireOpenDocument(document), sketch));
            return Segments(sketchObject).Select(SegmentItemOf).ToList();
        });

        /// <summary>Points of a sketch.</summary>
        public IReadOnlyList<PointItem> ListSketchPoints(DocumentRef document, string sketch) => Run("listing the sketch points", () =>
        {
            var sketchObject = SketchObject(SketchFeatureOf(RequireOpenDocument(document), sketch));
            return Points(sketchObject).Select(PointItemOf).ToList();
        });

        /// <summary>Geometric relations of a sketch.</summary>
        public IReadOnlyList<RelationItem> ListSketchRelations(DocumentRef document, string sketch) => Run("listing the sketch relations", () =>
        {
            var sketchObject = SketchObject(SketchFeatureOf(RequireOpenDocument(document), sketch));
            return Relations(sketchObject).Select(RelationItemOf).ToList();
        });

        /// <summary>Dimensions that belong to a sketch.</summary>
        public IReadOnlyList<DimensionItem> ListSketchDimensions(DocumentRef document, string sketch) => Run("listing the sketch dimensions", () =>
        {
            return DimensionReader.Describe(new[] { SketchFeatureOf(RequireOpenDocument(document), sketch) });
        });

        /// <summary>The sketch feature with that name, which must be a sketch (entity_not_found otherwise).</summary>
        private static SW.Feature SketchFeatureOf(SW.ModelDoc2 model, string name)
        {
            var feature = FeatureTree.Find(model, name);
            if (feature == null || !FeatureTree.IsSketch(feature))
            {
                throw ComHelper.NotFound("sketch", name);
            }

            return feature;
        }

        private static SW.Sketch SketchObject(SW.Feature feature) =>
            ComHelper.Require(feature.GetSpecificFeature2() as SW.Sketch, "the sketch of the feature");

        private static List<SW.SketchSegment> Segments(SW.Sketch sketch) =>
            ComWalk.Collection(sketch.GetSketchSegments(), "The sketch segment list").OfType<SW.SketchSegment>().ToList();

        private static List<SW.SketchPoint> Points(SW.Sketch sketch) =>
            ComWalk.Collection(sketch.GetSketchPoints2(), "The sketch point list").OfType<SW.SketchPoint>().ToList();

        private static List<SW.SketchRelation> Relations(SW.Sketch sketch) =>
            ComWalk.Collection(sketch.RelationManager.GetRelations((int)SWC.swSketchRelationFilterType_e.swAll), "The sketch relation list")
                .OfType<SW.SketchRelation>()
                .ToList();

        private static SegmentItem SegmentItemOf(SW.SketchSegment segment)
        {
            var type = SegmentType(segment);
            var item = new SegmentItem
            {
                Id = ComHelper.IdText(segment.GetID()),
                Type = type,
                Construction = segment.ConstructionGeometry,
                Length = segment.GetLength(),
            };
            ApplyGeometry(item, segment, type);
            return item;
        }

        private static string SegmentType(SW.SketchSegment segment)
        {
            switch ((SWC.swSketchSegments_e)segment.GetType())
            {
                case SWC.swSketchSegments_e.swSketchLINE:
                    return "line";
                case SWC.swSketchSegments_e.swSketchARC:
                    return IsCircle(segment) ? "circle" : "arc";
                case SWC.swSketchSegments_e.swSketchELLIPSE:
                    return "ellipse";
                case SWC.swSketchSegments_e.swSketchSPLINE:
                    return "spline";
                case SWC.swSketchSegments_e.swSketchTEXT:
                    return "text";
                case SWC.swSketchSegments_e.swSketchPARABOLA:
                    return "parabola";
                default:
                    return "other";
            }
        }

        private static bool IsCircle(SW.SketchSegment segment) =>
            segment is SW.SketchArc arc && arc.IsCircle() == 1;

        /// <summary>Start, end, centre and radius per segment type. Circles carry centre and radius only.</summary>
        private static void ApplyGeometry(SegmentItem item, SW.SketchSegment segment, string type)
        {
            switch (segment)
            {
                case SW.SketchLine line:
                    item.Start = Coordinates(() => line.GetStartPoint2());
                    item.End = Coordinates(() => line.GetEndPoint2());
                    break;
                case SW.SketchArc arc when type == "circle":
                    item.Center = Coordinates(() => arc.GetCenterPoint2());
                    item.Radius = ComHelper.TryValue(() => arc.GetRadius());
                    break;
                case SW.SketchArc arc:
                    item.Start = Coordinates(() => arc.GetStartPoint2());
                    item.End = Coordinates(() => arc.GetEndPoint2());
                    item.Center = Coordinates(() => arc.GetCenterPoint2());
                    item.Radius = ComHelper.TryValue(() => arc.GetRadius());
                    break;
                case SW.SketchEllipse ellipse:
                    item.Center = Coordinates(() => ellipse.GetCenterPoint2());
                    break;
                case SW.SketchSpline spline:
                    var controlPoints = Objects(() => spline.GetPoints2());
                    if (controlPoints.Count > 0)
                    {
                        item.Start = Coordinates(() => controlPoints[0]);
                        item.End = Coordinates(() => controlPoints[controlPoints.Count - 1]);
                    }

                    break;
            }
        }

        /// <summary>Optional geometry: the list, or empty when it cannot be read.</summary>
        private static List<object?> Objects(Func<object?> read) => ComHelper.Objects(ComHelper.TryObject(read));

        private static PointItem PointItemOf(SW.SketchPoint point) => new PointItem
        {
            Id = ComHelper.IdText(point.GetID()),
            Position = new[] { point.X, point.Y, point.Z },
            Construction = false,
        };

        private static RelationItem RelationItemOf(SW.SketchRelation relation)
        {
            var item = new RelationItem
            {
                Type = ReaderNames.ConstraintType(relation.GetRelationType()),
            };
            foreach (var entity in ComWalk.Collection(relation.GetEntities(), "The relation entity list"))
            {
                item.Entities.Add(new EntityRef { Kind = EntityKind(entity), Id = EntityId(entity) });
            }

            return item;
        }

        private static string EntityKind(object? entity) => entity switch
        {
            SW.SketchLine => "line",
            SW.SketchArc => "arc",
            SW.SketchPoint => "point",
            _ => "other",
        };

        private static string EntityId(object? entity) => entity switch
        {
            SW.SketchSegment segment => ComHelper.IdText(segment.GetID()),
            SW.SketchPoint point => ComHelper.IdText(point.GetID()),
            _ => string.Empty,
        };

        /// <summary>Coordinates [x, y, z] of a sketch point object, or null when unavailable (optional, nullable).</summary>
        private static double[]? Coordinates(Func<object?> read)
        {
            try
            {
                return read() is SW.SketchPoint point
                    ? new[] { point.X, point.Y, point.Z }
                    : null;
            }
            catch (Exception)
            {
                return null;
            }
        }

        /// <summary>The first reference plane or origin feature the sketch is built on, or null when it is on a face (optional).</summary>
        private static string? PlaneOrFace(SW.Feature sketchFeature) =>
            FeatureTree.Parents(sketchFeature)
                .FirstOrDefault(parent => IsPlaneType(FeatureTree.TypeName(parent)))
                is SW.Feature plane
                    ? FeatureTree.Name(plane)
                    : null;

        private static bool IsPlaneType(string typeName) =>
            string.Equals(typeName, "RefPlane", StringComparison.OrdinalIgnoreCase)
            || string.Equals(typeName, "OriginProfileFeature", StringComparison.OrdinalIgnoreCase);
    }
}
