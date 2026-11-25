using System;
using System.Runtime.InteropServices;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Pdm
{
    /// <summary>Builds the <see cref="PdmGatewayException"/> values the gateway throws.</summary>
    internal static class PdmErrors
    {
        /// <summary>
        /// HRESULT the PDM API returns when an item is not in the vault database (seen in spike S1 for a missing
        /// folder). Treated as "not found" by the lookups.
        /// </summary>
        private const int ItemNotInDatabase = unchecked((int)0x80040207);

        /// <summary>True for the HRESULT that means the item does not exist in the vault.</summary>
        public static bool IsNotFound(COMException error) => error.ErrorCode == ItemNotInDatabase;

        public static PdmGatewayException Api(string operation, COMException error) =>
            new PdmGatewayException(PdmFailureKind.Api, $"PDM could not {operation}: {error.Message}", null, error.ErrorCode, null, error);

        public static PdmGatewayException ClientMissing(Exception? inner = null) =>
            new PdmGatewayException(PdmFailureKind.ClientMissing, "The SOLIDWORKS PDM client is not installed on this computer.", null, null, null, inner);

        public static PdmGatewayException ViewMissing(string vault) =>
            new PdmGatewayException(PdmFailureKind.ViewMissing, $"This computer has no local view for PDM vault '{vault}'.", vault);

        public static PdmGatewayException LoginFailed(string vault, Exception? inner) =>
            new PdmGatewayException(PdmFailureKind.LoginFailed, $"Automatic login to PDM vault '{vault}' failed.", vault, null, null, inner);

        public static PdmGatewayException Busy() =>
            new PdmGatewayException(PdmFailureKind.Busy, "An earlier PDM request is still running. Try again later.");

        public static PdmGatewayException TimedOut() =>
            new PdmGatewayException(PdmFailureKind.Timeout, "The PDM request did not finish before its deadline.");

        public static PdmGatewayException NotFound(string vaultPath) =>
            new PdmGatewayException(PdmFailureKind.NotFound, $"Nothing exists at '{vaultPath}' in the PDM vault.", vaultPath);

        public static PdmGatewayException LinkInPath(string vaultPath) =>
            new PdmGatewayException(PdmFailureKind.LinkInPath, "The path goes through a link (symbolic link or junction).", vaultPath);

        public static PdmGatewayException InvalidSource(string sourcePath, string message) =>
            new PdmGatewayException(PdmFailureKind.InvalidSource, message, sourcePath);

        public static PdmGatewayException UnknownVariable(string name) =>
            new PdmGatewayException(PdmFailureKind.UnknownVariable, $"'{name}' is not a variable of this vault.", name);

        public static PdmGatewayException TooManyFiles(int maxFiles) =>
            new PdmGatewayException(PdmFailureKind.TooManyFiles, $"More than {maxFiles} files would be copied; nothing was copied.");

        /// <summary>A file or folder already exists on disk at the local path the add would write to. Nothing is overwritten.</summary>
        public static PdmGatewayException LocalFileExists(string targetVaultPath) =>
            new PdmGatewayException(PdmFailureKind.LocalFileExists, "A file already exists at the local path of the target. Nothing was changed.", targetVaultPath);
    }
}
