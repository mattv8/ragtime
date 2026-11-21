using System;
using System.Globalization;
using System.Runtime.InteropServices;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// Captures and restores the application state a script may change: document visibility per type, and in attach
    /// mode also UserControl and Visible. Runs on the instance's STA thread. A failed read or restore step is logged
    /// (pid, setting, exception type, HRESULT for COM errors) and the other steps still run. Script source is never logged.
    /// </summary>
    internal sealed partial class SolidWorksSession
    {
        private const string PartSetting = "DocumentVisible(part)";
        private const string AssemblySetting = "DocumentVisible(assembly)";
        private const string DrawingSetting = "DocumentVisible(drawing)";
        private const string UserControlSetting = "UserControl";
        private const string VisibleSetting = "Visible";

        /// <summary>Captures the state. Each read stands alone; a failed read is logged and leaves its field null.</summary>
        public ScriptSessionState CaptureScriptState()
        {
            var state = new ScriptSessionState
            {
                DocumentVisiblePart = TryRead(PartSetting, () => _application.GetDocumentVisible((int)SWC.swDocumentTypes_e.swDocPART)),
                DocumentVisibleAssembly = TryRead(AssemblySetting, () => _application.GetDocumentVisible((int)SWC.swDocumentTypes_e.swDocASSEMBLY)),
                DocumentVisibleDrawing = TryRead(DrawingSetting, () => _application.GetDocumentVisible((int)SWC.swDocumentTypes_e.swDocDRAWING)),
            };

            if (_options.Attached)
            {
                state.UserControl = TryRead(UserControlSetting, () => _application.UserControl);
                state.Visible = TryRead(VisibleSetting, () => _application.Visible);
            }

            var anyRead = state.DocumentVisiblePart.HasValue || state.DocumentVisibleAssembly.HasValue
                || state.DocumentVisibleDrawing.HasValue || state.UserControl.HasValue || state.Visible.HasValue;
            if (!anyRead)
            {
                throw HostError.Coded(
                    ErrorCodes.SolidworksUnavailable,
                    "SolidWorks did not report any of the state a script may change. The script was not run.");
            }

            return state;
        }

        /// <summary>
        /// Restores each captured field that has a value. Best effort: every step is tried, a failed step is logged, and
        /// nothing is thrown.
        /// </summary>
        public void RestoreScriptState(ScriptSessionState state)
        {
            if (state == null)
            {
                return;
            }

            RestoreDocumentVisible(state.DocumentVisiblePart, SWC.swDocumentTypes_e.swDocPART, PartSetting);
            RestoreDocumentVisible(state.DocumentVisibleAssembly, SWC.swDocumentTypes_e.swDocASSEMBLY, AssemblySetting);
            RestoreDocumentVisible(state.DocumentVisibleDrawing, SWC.swDocumentTypes_e.swDocDRAWING, DrawingSetting);

            if (_options.Attached)
            {
                RestoreStep(UserControlSetting, () => { if (state.UserControl.HasValue) _application.UserControl = state.UserControl.Value; });
                RestoreStep(VisibleSetting, () => { if (state.Visible.HasValue) _application.Visible = state.Visible.Value; });
            }
        }

        private void RestoreDocumentVisible(bool? visible, SWC.swDocumentTypes_e type, string setting)
        {
            if (!visible.HasValue)
            {
                return;
            }

            var value = visible.Value;
            RestoreStep(setting, () => _application.DocumentVisible(value, (int)type));
        }

        /// <summary>Reads one value. A COM failure is logged and gives null; no exception leaves this method.</summary>
        private T? TryRead<T>(string setting, Func<T> read) where T : struct
        {
            try
            {
                return read();
            }
            catch (Exception error) when (IsReadFailure(error))
            {
                LogFailure("capture", setting, error);
                return null;
            }
        }

        /// <summary>One restore step. Any failure is logged and the next step still runs; nothing is thrown.</summary>
        private void RestoreStep(string setting, Action step)
        {
            try
            {
                step();
            }
            catch (Exception error)
            {
                LogFailure("restore", setting, error);
            }
        }

        private static bool IsReadFailure(Exception error) =>
            error is COMException || error is InvalidComObjectException || error is InvalidCastException || error is NullReferenceException;

        /// <summary>
        /// One host log line: the action, pid, setting, exception type and, for COM errors, the HRESULT. The exception
        /// message and anything from the script are never written.
        /// </summary>
        private void LogFailure(string action, string setting, Exception error)
        {
            var line = "script state " + action + " failed pid=" + ProcessId.ToString(CultureInfo.InvariantCulture)
                + " setting=" + setting + " error=" + error.GetType().Name;
            if (error is COMException comError)
            {
                line += " hresult=0x" + comError.HResult.ToString("X8", CultureInfo.InvariantCulture);
            }

            try
            {
                _options.Log(line);
            }
            catch (Exception)
            {
                // A failing log writer must not break the restore or the capture.
            }
        }
    }
}
