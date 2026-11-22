using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using EPDM.Interop.epdm;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>Result of one look at the PDM client: its version and the local views it has. No login is involved.</summary>
    internal sealed class PdmProbe
    {
        public PdmProbe(string? version, IReadOnlyList<PdmViewFacts> views)
        {
            Version = version;
            Views = views;
        }

        public string? Version { get; }

        public IReadOnlyList<PdmViewFacts> Views { get; }
    }

    /// <summary>Client API calls that need no login. Runs on the PDM thread.</summary>
    internal static class PdmClientProbe
    {
        /// <summary>Reads the client version and views through a temporary vault object.</summary>
        public static PdmProbe Probe()
        {
            var client = CreateVault();
            try
            {
                return new PdmProbe(ReadVersion(client), ReadViews(client));
            }
            finally
            {
                Marshal.ReleaseComObject(client);
            }
        }

        /// <summary>Creates the PDM vault object. ClientMissing when the COM class cannot be activated.</summary>
        public static IEdmVault5 CreateVault()
        {
            try
            {
                return new EdmVault5();
            }
            catch (COMException error)
            {
                throw PdmErrors.ClientMissing(error);
            }
        }

        /// <summary>Local views of every vault on this computer. Login state is not part of the result.</summary>
        public static List<PdmViewFacts> ReadViews(IEdmVault5 client)
        {
            EdmViewInfo[]? views = null;
            // Window handle 0 and bOnlyLoggedIn false: no window is involved in listing the views.
            ((IEdmVault8)client).GetVaultViews(out views, false);
            var result = new List<PdmViewFacts>();
            if (views == null)
            {
                return result;
            }

            foreach (var view in views)
            {
                var vault = view.mbsVaultName ?? string.Empty;
                if (vault.Length == 0)
                {
                    continue;
                }

                result.Add(new PdmViewFacts(vault, (view.mbsPath ?? string.Empty).TrimEnd('\\'), false, null));
            }

            return result;
        }

        /// <summary>The client version as major.minor, or null when the client does not report it.</summary>
        public static string? ReadVersion(IEdmVault5 client)
        {
            try
            {
                int major = 0;
                int minor = 0;
                client.GetVersion(ref major, ref minor);
                return major + "." + minor;
            }
            catch (COMException)
            {
                return null;
            }
        }
    }
}
