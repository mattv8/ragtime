using Microsoft.Win32;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>Finds the SOLIDWORKS PDM client by its COM class registration. Reads the registry only.</summary>
    internal static class PdmClientRegistry
    {
        private const string ProgId = "ConisioLib.EdmVault";

        /// <summary>True when the PDM COM class has a CLSID in the 64-bit registry view.</summary>
        public static bool IsInstalled()
        {
            using (var classes = RegistryKey.OpenBaseKey(RegistryHive.ClassesRoot, RegistryView.Registry64))
            using (var clsid = classes.OpenSubKey(ProgId + @"\CLSID"))
            {
                return clsid?.GetValue(null) is string value && value.Length > 0;
            }
        }
    }
}
