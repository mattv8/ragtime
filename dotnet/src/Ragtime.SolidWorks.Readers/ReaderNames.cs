using System;
using Ragtime.SolidWorks.Core;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Maps SolidWorks enumeration values to the names used in the Core DTOs. Explicit switches are used where a
    /// SolidWorks name is not the readable name (dimension, tolerance, mate alignment) or where the enum has aliases.
    /// Other enums use the enum member name with its prefix removed, in lower snake case.
    /// </summary>
    internal static class ReaderNames
    {
        /// <summary>Document type from swDocumentTypes_e.</summary>
        public static string DocumentType(int value) => (SWC.swDocumentTypes_e)value switch
        {
            SWC.swDocumentTypes_e.swDocPART => ProtocolInfo.DocumentTypes.Part,
            SWC.swDocumentTypes_e.swDocASSEMBLY => ProtocolInfo.DocumentTypes.Assembly,
            SWC.swDocumentTypes_e.swDocDRAWING => ProtocolInfo.DocumentTypes.Drawing,
            _ => "other",
        };

        /// <summary>Constrained status of a sketch from swConstrainedStatus_e, e.g. under_constrained.</summary>
        public static string ConstrainedStatus(int value) =>
            EnumName(typeof(SWC.swConstrainedStatus_e), value, "sw") ?? "unknown";

        /// <summary>Sketch relation type from swConstraintType_e, e.g. horizontal.</summary>
        public static string ConstraintType(int value) =>
            EnumName(typeof(SWC.swConstraintType_e), value, "swConstraintType_") ?? "other";

        /// <summary>Mate type from swMateType_e, without the swMate prefix, e.g. coincident.</summary>
        public static string MateType(int value) =>
            EnumName(typeof(SWC.swMateType_e), value, "swMate") ?? "other";

        /// <summary>Mate alignment from swMateAlign_e.</summary>
        public static string MateAlignment(int value) => (SWC.swMateAlign_e)value switch
        {
            SWC.swMateAlign_e.swMateAlignALIGNED => "aligned",
            SWC.swMateAlign_e.swMateAlignANTI_ALIGNED => "anti_aligned",
            SWC.swMateAlign_e.swMateAlignCLOSEST => "closest",
            _ => "other",
        };

        /// <summary>Entity reference type of a mate, from swSelectType_e.</summary>
        public static string SelectType(int value) => (SWC.swSelectType_e)value switch
        {
            SWC.swSelectType_e.swSelEDGES => "edge",
            SWC.swSelectType_e.swSelFACES => "face",
            SWC.swSelectType_e.swSelVERTICES => "vertex",
            SWC.swSelectType_e.swSelDATUMPLANES => "plane",
            SWC.swSelectType_e.swSelDATUMAXES => "axis",
            SWC.swSelectType_e.swSelDATUMPOINTS => "point",
            SWC.swSelectType_e.swSelSKETCHSEGS => "sketch_segment",
            SWC.swSelectType_e.swSelSKETCHPOINTS => "sketch_point",
            SWC.swSelectType_e.swSelCOMPONENTS => "component",
            _ => "other",
        };

        /// <summary>Dimension type from swDimensionType_e: linear, angular, radial, diameter or other.</summary>
        public static string DimensionType(int value) => (SWC.swDimensionType_e)value switch
        {
            SWC.swDimensionType_e.swLinearDimension or SWC.swDimensionType_e.swHorLinearDimension
                or SWC.swDimensionType_e.swVertLinearDimension => "linear",
            SWC.swDimensionType_e.swAngularDimension => "angular",
            SWC.swDimensionType_e.swRadialDimension or SWC.swDimensionType_e.swRadialLinearDimension => "radial",
            SWC.swDimensionType_e.swDiameterDimension or SWC.swDimensionType_e.swDiametricLinearDimension => "diameter",
            _ => "other",
        };

        /// <summary>Tolerance type from swTolType_e. Callers skip swTolNONE.</summary>
        public static string ToleranceType(int value) => (SWC.swTolType_e)value switch
        {
            SWC.swTolType_e.swTolBASIC => "basic",
            SWC.swTolType_e.swTolBILAT => "bilateral",
            SWC.swTolType_e.swTolLIMIT => "limit",
            SWC.swTolType_e.swTolSYMMETRIC => "symmetric",
            SWC.swTolType_e.swTolMIN => "min",
            SWC.swTolType_e.swTolMAX => "max",
            SWC.swTolType_e.swTolFIT => "fit",
            SWC.swTolType_e.swTolFITWITHTOL => "fit_with_tolerance",
            SWC.swTolType_e.swTolFITTOLONLY => "fit_tolerance_only",
            SWC.swTolType_e.swTolBLOCK => "block",
            SWC.swTolType_e.swTolGeneral => "general",
            _ => "other",
        };

        /// <summary>Custom property value type from swCustomInfoType_e.</summary>
        public static string CustomInfoType(int value) => (SWC.swCustomInfoType_e)value switch
        {
            SWC.swCustomInfoType_e.swCustomInfoText => "text",
            SWC.swCustomInfoType_e.swCustomInfoNumber => "number",
            SWC.swCustomInfoType_e.swCustomInfoDouble => "double",
            SWC.swCustomInfoType_e.swCustomInfoYesOrNo => "yes_or_no",
            SWC.swCustomInfoType_e.swCustomInfoDate => "date",
            _ => "other",
        };

        private static string? EnumName(Type enumType, int value, string prefix)
        {
            var name = Enum.GetName(enumType, value);
            return name == null ? null : ComHelper.Snake(ComHelper.StripPrefix(name, prefix));
        }
    }
}
