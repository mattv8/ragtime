using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>
    /// The startup sequences, run on an instance's STA thread, and the close helpers used on exit.
    /// <list type="bullet">
    /// <item><see cref="RunOwned"/>: a host-launched process. Waits for SolidWorks to finish starting, then makes it
    /// invisible (before the instance is ready, so no document is ever opened in a visible window).</item>
    /// <item><see cref="RunAttach"/>: the readiness operation of a SolidWorks the user started. It runs until the
    /// instance is ready or the process is gone; the caller only waits a bounded time for it (see the manager). Never
    /// changes the window or the user's configuration.</item>
    /// </list>
    /// </summary>
    internal static class InstanceStartup
    {
        private const int PollMilliseconds = 250;
        private const int MaxDocumentsToList = 10000;

        /// <summary>Startup of a host-launched instance. Returns true when ready; throws HostErrorException otherwise.</summary>
        public static bool RunOwned(
            InstanceRecord record,
            Deadline deadline,
            ISolidWorksSessionFactory sessionFactory,
            IMonotonicClock clock,
            Action<string> log)
        {
            var started = clock.ElapsedMilliseconds;
            var application = WaitForApplication(record, deadline);
            record.App = application;

            WaitForReady(record, application, deadline);
            ConfigureBackground(record, application, log);

            var revision = ReadRevision(application, ErrorCodes.SolidworksStartFailed);
            CheckRevision(record, revision, log);

            // Re-hiding runs on this instance's STA thread: the session calls it from inside a work item.
            var options = new SessionOptions(false, () => { EnsureHidden(record, application, log); }, log);
            var session = CreateSession(sessionFactory, application, ErrorCodes.SolidworksStartFailed, options);
            var visible = EnsureHidden(record, application, log);

            Publish(record, session, revision, record.Year, visible, log, clock.ElapsedMilliseconds - started);
            return true;
        }

        /// <summary>
        /// Readiness of an attached SolidWorks: waits (polling, on the instance's STA thread) until the application is
        /// registered, has finished starting and answers its revision. Returns when ready. Throws when the process goes
        /// away (solidworks_not_running); a blocked COM call simply keeps this operation outstanding.
        /// </summary>
        public static bool RunAttach(
            InstanceRecord record,
            ISolidWorksSessionFactory sessionFactory,
            IMonotonicClock clock,
            Action<string> log)
        {
            var started = clock.ElapsedMilliseconds;
            var (application, revision) = WaitForAttachReady(record);
            record.App = application;

            var year = SolidWorksVersions.RevisionToYear(revision);
            // Attached instances are never hidden by the host: re-hiding is a no-op.
            var options = new SessionOptions(true, () => { }, log);
            var session = CreateSession(sessionFactory, application, ErrorCodes.SolidworksError, options);
            var visible = ReadVisible(application) ?? false;

            Publish(record, session, revision, year, visible, log, clock.ElapsedMilliseconds - started);
            return true;
        }

        /// <summary>Reads the visibility of the application; null when COM could not answer.</summary>
        public static bool? ReadVisible(SW.ISldWorks application)
        {
            try
            {
                return application.Visible;
            }
            catch (COMException)
            {
                return null;
            }
            catch (InvalidComObjectException)
            {
                return null;
            }
        }

        /// <summary>
        /// Owned instances only, on the STA thread, after the background settings and after every completed call.
        /// Hides the application's frame window (COM handle, when available) and every visible titled top-level window
        /// of the process. Logs each hide as one line. Tracks the close-rule state: a titled window seen after the first
        /// hide sets <see cref="InstanceRecord.EverVisibleAfterHide"/>. Returns true when a titled top-level window of
        /// the process is still visible after the hide attempt. Never called for attached instances.
        /// </summary>
        public static bool EnsureHidden(InstanceRecord record, SW.ISldWorks application, Action<string> log)
        {
            try
            {
                var frame = FrameHandle(application);
                if (frame != IntPtr.Zero)
                {
                    WindowControl.Hide(frame);
                }

                var titled = WindowControl.VisibleTitledWindows(record.ProcessId);
                if (titled.Count > 0)
                {
                    if (record.FirstHideDone)
                    {
                        record.EverVisibleAfterHide = true;
                    }

                    foreach (var window in titled)
                    {
                        WindowControl.Hide(window);
                    }

                    record.WindowHideCount++;
                    log(record.WindowHideCount == 1
                        ? $"{record.Id}: hid {titled.Count} visible window(s) of pid {record.ProcessId} (first hide)."
                        : $"{record.Id}: a window became visible again; hid {titled.Count} (hide {record.WindowHideCount}).");
                }

                var remaining = WindowControl.VisibleTitledWindows(record.ProcessId).Count;
                if (remaining > 0)
                {
                    // Still visible after the host hid it: a user could have edited in it.
                    record.EverVisibleAfterHide = true;
                }

                record.FirstHideDone = true;
                return remaining > 0;
            }
            catch (Exception error)
            {
                // Unknown state is reported as visible: the snapshot must not claim a hidden window it could not check.
                record.EverVisibleAfterHide = true;
                log($"{record.Id}: window check failed ({error.GetType().Name}); reported as visible.");
                return true;
            }
        }

        /// <summary>
        /// Right before a close decision (owned instances): a titled window that is visible now, after the host's first
        /// hide, sets <see cref="InstanceRecord.EverVisibleAfterHide"/>. An enumeration that fails counts as visible.
        /// </summary>
        public static void NoteWindowsBeforeClose(InstanceRecord record)
        {
            try
            {
                if (record.FirstHideDone && WindowControl.VisibleTitledWindows(record.ProcessId).Count > 0)
                {
                    record.EverVisibleAfterHide = true;
                }
            }
            catch (Exception)
            {
                record.EverVisibleAfterHide = true;
            }
        }

        /// <summary>
        /// Closes every document without saving and asks SolidWorks to exit, on the STA thread. Clears the session and
        /// releases the application reference before returning, so the caller can wait for the process to exit.
        /// </summary>
        public static void CloseAndExit(InstanceRecord record, Action<string> log)
        {
            var application = record.App;
            if (application != null)
            {
                try
                {
                    // Documents are only ever opened read-only, so discarding changes is safe once they were checked.
                    application.CloseAllDocuments(true);
                    log($"close {record.Id}: documents closed without saving");
                }
                catch (COMException error)
                {
                    log($"close {record.Id}: CloseAllDocuments failed ({HexOf(error)})");
                }

                try
                {
                    application.ExitApp();
                    log($"close {record.Id}: exit requested");
                }
                catch (COMException error)
                {
                    log($"close {record.Id}: ExitApp failed ({HexOf(error)})");
                }
            }

            ReleaseSessionAndApplication(record, log);
        }

        /// <summary>
        /// Lists the paths of the open documents that have unsaved changes, on the STA thread. Returns null when the
        /// documents could not be listed. An instance without an application has no open documents.
        /// </summary>
        public static IReadOnlyList<string>? UnsavedDocumentPaths(InstanceRecord record, Action<string> log) =>
            ListDocumentPaths(record, log, unsavedOnly: true)?.Select(path => path ?? string.Empty).ToList();

        /// <summary>
        /// Lists the paths of every open document, on the STA thread. Returns null when the documents could not be listed.
        /// A document whose path cannot be read gives a null entry, so the copy retention sweep fails closed (review finding 3);
        /// a document with no path (unsaved) gives an empty string.
        /// </summary>
        public static IReadOnlyList<string?>? OpenDocumentPaths(InstanceRecord record, Action<string> log) =>
            ListDocumentPaths(record, log, unsavedOnly: false);

        private static IReadOnlyList<string?>? ListDocumentPaths(InstanceRecord record, Action<string> log, bool unsavedOnly)
        {
            var application = record.App;
            if (application == null)
            {
                return new List<string>();
            }

            var paths = new List<string?>();
            SW.ModelDoc2? document = null;
            try
            {
                document = application.GetFirstDocument() as SW.ModelDoc2;
                var listed = 0;
                while (document != null)
                {
                    if (++listed > MaxDocumentsToList)
                    {
                        log($"{record.Id}: more than {MaxDocumentsToList} open documents; cannot check them all.");
                        return null;
                    }

                    if (unsavedOnly)
                    {
                        if (SafeSaveFlag(document, log))
                        {
                            paths.Add(SafePath(document));
                        }
                    }
                    else
                    {
                        paths.Add(OpenPathOf(document));
                    }

                    var next = document.GetNext() as SW.ModelDoc2;
                    Release(document);
                    document = next;
                }

                return paths;
            }
            catch (COMException error)
            {
                log($"{record.Id}: could not list open documents ({HexOf(error)})");
                return null;
            }
            finally
            {
                if (document != null)
                {
                    Release(document);
                }
            }
        }

        /// <summary>Clears the session and releases the application's COM reference. Runs on the STA thread.</summary>
        public static void ReleaseSessionAndApplication(InstanceRecord record, Action<string> log)
        {
            record.Session = null;
            ReleaseApplication(record);
            log($"{record.Id}: COM reference released");
        }

        /// <summary>Releases the application's COM reference. Runs on the instance's STA thread when the worker exits.</summary>
        public static void ReleaseApplication(InstanceRecord record)
        {
            var application = record.App;
            record.App = null;
            if (application == null)
            {
                return;
            }

            try
            {
                Marshal.ReleaseComObject(application);
            }
            catch (InvalidComObjectException)
            {
                // Already released; nothing left to do.
            }
        }

        /// <summary>Hex form of a COM HRESULT for details and logs.</summary>
        public static string HexOf(COMException error) =>
            "0x" + error.ErrorCode.ToString("X8", CultureInfo.InvariantCulture);

        private static SW.ISldWorks WaitForApplication(InstanceRecord record, Deadline deadline)
        {
            while (true)
            {
                RequireRunning(record);
                var application = TryFind(record.ProcessId);
                if (application != null)
                {
                    return application;
                }

                RequireTime(record, deadline, "did not register with the Running Object Table");
                Thread.Sleep(PollMilliseconds);
            }
        }

        private static void WaitForReady(InstanceRecord record, SW.ISldWorks application, Deadline deadline)
        {
            while (true)
            {
                RequireRunning(record);
                if (IsStartupCompleted(application))
                {
                    return;
                }

                RequireTime(record, deadline, "did not finish starting");
                Thread.Sleep(PollMilliseconds);
            }
        }

        /// <summary>
        /// Polls until the application is registered, has finished starting and answers its revision. Only a process
        /// that goes away ends the wait with an error; COM refusals are retried.
        /// </summary>
        private static (SW.ISldWorks Application, string? Revision) WaitForAttachReady(InstanceRecord record)
        {
            while (true)
            {
                RequireRunning(record);
                var application = TryFind(record.ProcessId);
                if (application != null)
                {
                    if (IsStartupCompleted(application) && TryReadRevision(application, out var revision))
                    {
                        return (application, revision);
                    }

                    Release(application);
                }

                Thread.Sleep(PollMilliseconds);
            }
        }

        private static bool TryReadRevision(SW.ISldWorks application, out string? revision)
        {
            try
            {
                revision = application.RevisionNumber();
                return true;
            }
            catch (COMException)
            {
                revision = null;
                return false;
            }
        }

        private static SW.ISldWorks? TryFind(int processId)
        {
            try
            {
                return RunningObjectTable.FindApplication(processId);
            }
            catch (COMException)
            {
                // Not answering yet; the caller polls again.
                return null;
            }
        }

        private static bool IsStartupCompleted(SW.ISldWorks application)
        {
            try
            {
                return application.StartupProcessCompleted;
            }
            catch (COMException)
            {
                // Calls are rejected while SolidWorks starts; the caller polls again.
                return false;
            }
        }

        /// <summary>
        /// The settings that keep a host-launched instance out of sight, applied once before the instance is ready
        /// (so before any document is opened). Each step runs on its own: a failed step is logged and the next one runs.
        /// </summary>
        private static void ConfigureBackground(InstanceRecord record, SW.ISldWorks application, Action<string> log)
        {
            Attempt(record, log, "UserControl = false", () => application.UserControl = false);
            Attempt(record, log, "UserControlBackground = true", () => application.UserControlBackground = true);
            Attempt(record, log, "Visible = false", () => application.Visible = false);
            Attempt(record, log, "Frame.KeepInvisible = true", () => ((SW.Frame)application.Frame()).KeepInvisible = true);
            Attempt(record, log, "DocumentVisible(false, part)", () => application.DocumentVisible(false, (int)SWC.swDocumentTypes_e.swDocPART));
            Attempt(record, log, "DocumentVisible(false, assembly)", () => application.DocumentVisible(false, (int)SWC.swDocumentTypes_e.swDocASSEMBLY));
            Attempt(record, log, "DocumentVisible(false, drawing)", () => application.DocumentVisible(false, (int)SWC.swDocumentTypes_e.swDocDRAWING));
            log($"{record.Id}: background settings applied (no user control, hidden).");
        }

        /// <summary>
        /// The application's main frame window handle, or zero when COM does not provide it (the window enumeration
        /// still covers the frame).
        /// </summary>
        private static IntPtr FrameHandle(SW.ISldWorks application)
        {
            try
            {
                var frame = (SW.Frame)application.Frame();
                return new IntPtr(frame.GetHWndx64());
            }
            catch (Exception)
            {
                return IntPtr.Zero;
            }
        }

        /// <summary>Runs one COM setting; a failure is logged with its HRESULT and does not stop the startup.</summary>
        private static void Attempt(InstanceRecord record, Action<string> log, string step, Action action)
        {
            try
            {
                action();
            }
            catch (COMException error)
            {
                log($"{record.Id}: {step} failed ({HexOf(error)}).");
            }
            catch (Exception error)
            {
                log($"{record.Id}: {step} failed ({error.GetType().Name}).");
            }
        }

        private static string? ReadRevision(SW.ISldWorks application, string failureCode)
        {
            try
            {
                return application.RevisionNumber();
            }
            catch (COMException error)
            {
                throw HostError.Coded(
                    failureCode,
                    "SolidWorks did not report its revision number.",
                    new JObject { ["hresult"] = WireJson.Value(HexOf(error)) });
            }
        }

        private static void CheckRevision(InstanceRecord record, string? revision, Action<string> log)
        {
            var revisionYear = SolidWorksVersions.RevisionToYear(revision);
            if (revisionYear != record.Year)
            {
                log($"{record.Id}: revision {revision ?? "unknown"} does not match the expected year {record.Year}.");
            }
        }

        private static ISolidWorksSession CreateSession(ISolidWorksSessionFactory factory, SW.ISldWorks application, string failureCode, SessionOptions options)
        {
            try
            {
                return factory.Create(application, options);
            }
            catch (HostErrorException)
            {
                throw;
            }
            catch (Exception error)
            {
                throw HostError.Coded(
                    failureCode,
                    "The SolidWorks agent could not create its session for this SolidWorks instance.",
                    new JObject { ["exception_type"] = WireJson.Value(error.GetType().Name) });
            }
        }

        private static void Publish(
            InstanceRecord record,
            ISolidWorksSession session,
            string? revision,
            int? year,
            bool visible,
            Action<string> log,
            long elapsedMs)
        {
            lock (record.Sync)
            {
                record.Session = session;
                record.Revision = revision;
                record.Year = year;
                record.Visible = visible;
                if (record.State == InstanceState.Starting)
                {
                    record.State = InstanceState.Ready;
                }
            }

            log($"ready {record.Id}: SolidWorks {VersionText(year)} pid {record.ProcessId} after {elapsedMs} ms ({(record.Owned ? "background" : "attached")})");
        }

        private static string VersionText(int? year) =>
            year.HasValue ? year.Value.ToString(CultureInfo.InvariantCulture) : "unknown";

        private static void RequireRunning(InstanceRecord record)
        {
            if (record.Retired)
            {
                throw HostError.Coded(
                    ErrorCodes.SolidworksNotRunning,
                    $"SolidWorks instance {record.Id} was stopped while it was starting.");
            }

            if (ProcessControl.IsGone(record.ProcessObject))
            {
                var code = record.Owned ? ErrorCodes.SolidworksStartFailed : ErrorCodes.SolidworksNotRunning;
                throw HostError.Coded(
                    code,
                    $"SolidWorks instance {record.Id} exited before it finished starting.",
                    new JObject { ["pid"] = WireJson.Value(record.ProcessId) });
            }
        }

        private static void RequireTime(InstanceRecord record, Deadline deadline, string problem)
        {
            if (deadline.Expired)
            {
                throw HostError.Coded(
                    ErrorCodes.InstanceTimeout,
                    $"SolidWorks instance {record.Id} {problem} within {(int)deadline.Total.TotalSeconds} seconds.",
                    new JObject { ["instance_id"] = WireJson.Value(record.Id) });
            }
        }

        private static bool SafeSaveFlag(SW.ModelDoc2 document, Action<string> log)
        {
            try
            {
                return document.GetSaveFlag();
            }
            catch (COMException error)
            {
                // An unreadable flag counts as unsaved: the document is not closed without a check.
                log($"save state unreadable ({HexOf(error)}); treated as unsaved");
                return true;
            }
        }

        private static string SafePath(SW.ModelDoc2 document)
        {
            try
            {
                return document.GetPathName() ?? "(unsaved document)";
            }
            catch (COMException)
            {
                return "(unnamed document)";
            }
        }

        /// <summary>
        /// The path of an open document for the retention sweep: empty when the document has no path, null when the path
        /// cannot be read (the sweep then deletes nothing).
        /// </summary>
        private static string? OpenPathOf(SW.ModelDoc2 document)
        {
            try
            {
                return document.GetPathName() ?? string.Empty;
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static void Release(object comObject)
        {
            try
            {
                Marshal.ReleaseComObject(comObject);
            }
            catch (InvalidComObjectException)
            {
                // Already released.
            }
        }
    }
}
