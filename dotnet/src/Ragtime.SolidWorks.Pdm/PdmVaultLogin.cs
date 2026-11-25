using System;
using System.Runtime.InteropServices;
using EPDM.Interop.epdm;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>Opens a logged-in session on the PDM thread: view check, then the machine's automatic login.</summary>
    internal static class PdmVaultLogin
    {
        /// <summary>Returns a session for the named vault. ViewMissing, LoginFailed or ClientMissing otherwise.</summary>
        public static PdmVault Open(string vaultName, Action<string> log)
        {
            var client = PdmClientProbe.CreateVault();
            try
            {
                return OpenWith(client, vaultName, log);
            }
            catch
            {
                Marshal.ReleaseComObject(client);
                throw;
            }
        }

        private static PdmVault OpenWith(IEdmVault5 client, string vaultName, Action<string> log)
        {
            System.Collections.Generic.List<Ragtime.SolidWorks.Core.PdmViewFacts> views;
            try { views = PdmClientProbe.ReadViews(client); }
            catch (COMException error) { throw PdmErrors.Api("read vault views", error); }
            var view = views.Find(candidate => string.Equals(candidate.Vault, vaultName, StringComparison.OrdinalIgnoreCase));
            if (view == null)
            {
                throw PdmErrors.ViewMissing(vaultName);
            }

            try
            {
                // Window handle 0: the automatic login may not show a dialog from this thread.
                client.LoginAuto(view.Vault, 0);
                if (!client.IsLoggedIn)
                {
                    throw PdmErrors.LoginFailed(view.Vault, null);
                }

                var user = ((IEdmUserMgr5)client).GetLoggedInUser();
                return new PdmVault(client, view.Vault, view.Root, user.Name, user.ID, log);
            }
            catch (COMException error)
            {
                throw PdmErrors.LoginFailed(view.Vault, error);
            }
        }
    }
}
