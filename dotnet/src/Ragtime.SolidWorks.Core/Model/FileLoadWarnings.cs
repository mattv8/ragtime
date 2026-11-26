using System.Collections.Generic;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>
    /// Names for the <c>swFileLoadWarning_e</c> bits that <c>ISldWorks::OpenDoc6</c> reports (contract §5.5). The session
    /// passes the raw bitmask; Core turns it into the <c>load_warnings</c> names. A bit this table does not name is
    /// reported as "unknown", once. Wire contract: the chat guidance quotes these names.
    /// </summary>
    public static class FileLoadWarnings
    {
        /// <summary>Name reported for a set bit that the table does not know.</summary>
        public const string UnknownName = "unknown";

        private static readonly (int Bit, string Name)[] Table =
        {
            (0x1, "id_mismatch"),
            (0x2, "read_only"),
            (0x4, "sharing_violation"),
            (0x8, "drawing_ansi_update"),
            (0x10, "sheet_scale_update"),
            (0x20, "needs_regen"),
            (0x40, "base_part_not_loaded"),
            (0x80, "already_open"),
            (0x100, "drawings_only_rapid_draft"),
            (0x200, "view_only_restrictions"),
            (0x400, "view_missing_referenced_config"),
            (0x800, "drawing_sf_symbol_convert"),
            (0x1000, "revolve_dim_tolerance"),
            (0x2000, "model_out_of_date"),
            (0x4000, "dimensions_referenced_incorrectly"),
            (0x8000, "component_missing_referenced_config"),
            (0x10000, "linked_design_table_update_failed"),
            (0x20000, "missing_design_table"),
            (0x40000, "automatic_repair"),
            (0x80000, "critical_data_repair"),
            (0x100000, "references_not_found"),
        };

        /// <summary>Names of the bits set in <paramref name="bits"/>, in bit order; an empty list when none is set.</summary>
        public static IReadOnlyList<string> NamesOf(int bits)
        {
            var names = new List<string>();
            var remaining = bits;
            foreach (var (bit, name) in Table)
            {
                if ((bits & bit) != 0)
                {
                    names.Add(name);
                    remaining &= ~bit;
                }
            }

            if (remaining != 0)
            {
                names.Add(UnknownName);
            }

            return names;
        }
    }
}
