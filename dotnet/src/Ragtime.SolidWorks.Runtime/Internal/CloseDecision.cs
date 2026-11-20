using System.Collections.Generic;

namespace Ragtime.SolidWorks.Runtime
{
    /// <summary>What the host knows about unsaved work in one owned instance, decided on its STA thread.</summary>
    internal enum SaveState
    {
        /// <summary>The documents were listed and none has unsaved changes.</summary>
        Clean,

        /// <summary>At least one open document has unsaved changes.</summary>
        Unsaved,

        /// <summary>The documents could not be listed in time, or listing failed.</summary>
        Unknown,
    }

    /// <summary>Result of the save-state check before a close: the state, and the unsaved document paths or the reason.</summary>
    internal sealed class SaveCheck
    {
        private SaveCheck(SaveState state, IReadOnlyList<string> unsaved, string reason)
        {
            State = state;
            Unsaved = unsaved;
            Reason = reason;
        }

        public SaveState State { get; }

        /// <summary>Paths of the unsaved documents (only for <see cref="SaveState.Unsaved"/>).</summary>
        public IReadOnlyList<string> Unsaved { get; }

        /// <summary>Why the state is unknown (only for <see cref="SaveState.Unknown"/>).</summary>
        public string Reason { get; }

        public static SaveCheck Clean() => new SaveCheck(SaveState.Clean, new List<string>(), string.Empty);

        public static SaveCheck WithUnsaved(IReadOnlyList<string> paths) => new SaveCheck(SaveState.Unsaved, paths, string.Empty);

        public static SaveCheck Unknown(string reason) => new SaveCheck(SaveState.Unknown, new List<string>(), reason);
    }

    /// <summary>
    /// The unsaved-work rule (ruling F1) as a pure decision. Only a known clean state, or an unknown state with no
    /// titled window ever seen after the host's first hide (nobody could have edited in it), allows a close. A known
    /// unsaved state, or an unknown state after a visible window, releases the instance: no close, no kill.
    /// </summary>
    internal static class CloseDecision
    {
        /// <summary>True when the instance may be closed (gracefully, then killed after the grace period if needed).</summary>
        public static bool MayClose(SaveState state, bool everVisibleAfterHide) =>
            state == SaveState.Clean
            || (state == SaveState.Unknown && !everVisibleAfterHide);
    }
}
