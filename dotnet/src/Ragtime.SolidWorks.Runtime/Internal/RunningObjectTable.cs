using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using SW = global::SolidWorks.Interop.sldworks;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>
    /// Finds a SolidWorks application object in the Running Object Table by its moniker
    /// (display name contains "SolidWorks_PID_&lt;pid&gt;"). Must run on an STA thread.
    /// </summary>
    internal static class RunningObjectTable
    {
        private const string MonikerPrefix = "SolidWorks_PID_";

        [DllImport("ole32.dll")]
        private static extern int CreateBindCtx(int reserved, out IBindCtx bindContext);

        [DllImport("ole32.dll")]
        private static extern int GetRunningObjectTable(int reserved, out IRunningObjectTable table);

        /// <summary>The application object of the process, or null while it is not registered.</summary>
        public static SW.ISldWorks? FindApplication(int processId)
        {
            var token = MonikerPrefix + processId.ToString(CultureInfo.InvariantCulture);
            IBindCtx? bindContext = null;
            IRunningObjectTable? table = null;
            IEnumMoniker? monikers = null;
            try
            {
                Marshal.ThrowExceptionForHR(CreateBindCtx(0, out bindContext));
                Marshal.ThrowExceptionForHR(GetRunningObjectTable(0, out table));
                table.EnumRunning(out monikers);

                var buffer = new IMoniker[1];
                while (monikers.Next(1, buffer, IntPtr.Zero) == 0)
                {
                    var moniker = buffer[0];
                    try
                    {
                        if (!TryGetDisplayName(bindContext, moniker, out var name) || !MatchesToken(name, token))
                        {
                            continue;
                        }

                        object? found;
                        table.GetObject(moniker, out found);
                        return found as SW.ISldWorks;
                    }
                    finally
                    {
                        Marshal.ReleaseComObject(moniker);
                    }
                }

                return null;
            }
            catch (COMException)
            {
                return null;
            }
            finally
            {
                Release(monikers);
                Release(table);
                Release(bindContext);
            }
        }

        private static bool TryGetDisplayName(IBindCtx bindContext, IMoniker moniker, out string name)
        {
            try
            {
                moniker.GetDisplayName(bindContext, null, out name);
                return name != null;
            }
            catch (COMException)
            {
                name = string.Empty;
                return false;
            }
        }

        /// <summary>True when the name holds the token and the token is not a prefix of a longer process id.</summary>
        private static bool MatchesToken(string name, string token)
        {
            var index = name.IndexOf(token, StringComparison.OrdinalIgnoreCase);
            if (index < 0)
            {
                return false;
            }

            var end = index + token.Length;
            return end == name.Length || !char.IsDigit(name[end]);
        }

        private static void Release(object? comObject)
        {
            if (comObject != null)
            {
                Marshal.ReleaseComObject(comObject);
            }
        }
    }
}
