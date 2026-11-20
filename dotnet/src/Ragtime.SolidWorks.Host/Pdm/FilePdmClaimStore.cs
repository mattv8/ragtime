using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>
    /// PDM claims kept in %LOCALAPPDATA%\RagtimeSolidWorks\pdm-claims.txt, one JSON object per line with every
    /// <see cref="PdmClaim"/> field (<c>ClaimedUtc</c> as ISO 8601 UTC). Loaded on construction; every change rewrites
    /// the whole file through a temporary file. Thread-safe.
    /// <para>
    /// <see cref="Put"/> never throws: a failed write is logged and the claim stays in memory. After a restart such a
    /// claim is gone, which is the safe direction. <see cref="Remove"/> must be durable: the file is rewritten (with one
    /// immediate retry), and when that fails the claim is restored in memory, the failure is logged and an
    /// <see cref="IOException"/> is thrown.
    /// </para>
    /// <para>
    /// <c>_diskStale</c> is set when the file may differ from memory (a failed write, or duplicate or dropped records
    /// found on load). While it is set, <see cref="Remove"/> rewrites the file even for a claim that is not in memory,
    /// so that a removed claim cannot come back after a restart. Lines that do not parse, including the old
    /// <c>vault|fileId</c> format, are skipped; they carry no claim. Vault names compare ordinal ignore-case. At most
    /// <see cref="MaxEntries"/> claims: the oldest <c>ClaimedUtc</c> is dropped.
    /// </para>
    /// </summary>
    internal sealed class FilePdmClaimStore : IPdmClaimStore
    {
        /// <summary>Claims kept; the oldest are dropped beyond this.</summary>
        public const int MaxEntries = 10000;

        private const string RelativePath = @"RagtimeSolidWorks\pdm-claims.txt";
        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false);
        private readonly object _gate = new object();
        private readonly string _path;
        private readonly Action<string> _log;
        private readonly Dictionary<string, PdmClaim> _claims = new Dictionary<string, PdmClaim>(StringComparer.OrdinalIgnoreCase);
        private bool _diskStale;

        /// <summary>Loads the claims at <paramref name="path"/>. Never throws.</summary>
        public FilePdmClaimStore(string path, Action<string> log)
        {
            _path = path ?? throw new ArgumentNullException(nameof(path));
            _log = log ?? throw new ArgumentNullException(nameof(log));
            Load();
        }

        /// <summary>The store for the current user; an in-memory store when the location cannot be resolved. Never throws.</summary>
        public static IPdmClaimStore ForCurrentUser(Action<string> log)
        {
            try
            {
                var root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                if (!string.IsNullOrEmpty(root))
                {
                    return new FilePdmClaimStore(Path.Combine(root, RelativePath), log);
                }
            }
            catch (Exception exception)
            {
                log("PDM claims: the local application data folder is unavailable; claims are kept in memory. " + exception.Message);
                return new InMemoryPdmClaimStore();
            }

            log("PDM claims: the local application data folder is unavailable; claims are kept in memory.");
            return new InMemoryPdmClaimStore();
        }

        /// <inheritdoc />
        public PdmClaim? Find(string vault, int fileId)
        {
            lock (_gate)
            {
                return _claims.TryGetValue(Key(vault, fileId), out var claim) ? Copy(claim) : null;
            }
        }

        /// <inheritdoc />
        /// <remarks>Never throws. A failed write is logged; the claim stays in memory.</remarks>
        public void Put(PdmClaim claim)
        {
            if (claim == null)
            {
                throw new ArgumentNullException(nameof(claim));
            }

            lock (_gate)
            {
                _claims[Key(claim.Vault, claim.FileId)] = Copy(claim);
                DropOldest();
                TryRewrite();
            }
        }

        /// <inheritdoc />
        /// <remarks>
        /// Durable or throws. Throws <see cref="IOException"/> when the file cannot be rewritten after one retry; the claim
        /// is then still in memory.
        /// </remarks>
        public void Remove(string vault, int fileId)
        {
            lock (_gate)
            {
                var key = Key(vault, fileId);
                if (!_claims.TryGetValue(key, out var claim) && !_diskStale)
                {
                    return;
                }

                _claims.Remove(key);
                if (TryRewrite() || TryRewrite())
                {
                    return;
                }

                if (claim != null)
                {
                    _claims[key] = claim;
                }

                _log("PDM claims: the removal could not be written; the claim stays in memory and will be removed again on the next attempt.");
                throw new IOException("The PDM claim removal could not be written to " + _path + ".");
            }
        }

        /// <summary>Drops the oldest claims beyond the cap from memory; returns how many were dropped.</summary>
        private int DropOldest()
        {
            var dropped = 0;
            while (_claims.Count > MaxEntries)
            {
                var oldest = _claims.Values.OrderBy(claim => claim.ClaimedUtc).First();
                _claims.Remove(Key(oldest.Vault, oldest.FileId));
                dropped++;
            }

            return dropped;
        }

        private void Load()
        {
            if (!File.Exists(_path))
            {
                return;
            }

            string[] lines;
            try
            {
                lines = File.ReadAllLines(_path, Utf8);
            }
            catch (Exception exception)
            {
                _log("PDM claims: the claims file could not be read; starting with no claims. " + exception.Message);
                return;
            }

            var skipped = 0;
            var duplicates = 0;
            foreach (var line in lines)
            {
                if (line.Trim().Length == 0)
                {
                    continue;
                }

                var claim = TryParse(line);
                if (claim == null)
                {
                    skipped++;
                    continue;
                }

                var key = Key(claim.Vault, claim.FileId);
                if (_claims.ContainsKey(key))
                {
                    duplicates++;
                }

                _claims[key] = claim;
            }

            // The file then holds records memory does not, so a later Remove must rewrite it.
            var dropped = DropOldest();
            _diskStale = duplicates > 0 || dropped > 0;
            if (skipped > 0)
            {
                _log("PDM claims: skipped " + skipped.ToString(CultureInfo.InvariantCulture) + " unreadable line(s) in the claims file.");
            }
        }

        /// <summary>Rewrites the whole file from memory. Returns false and logs on failure; never throws.</summary>
        private bool TryRewrite()
        {
            var temporary = _path + ".tmp";
            try
            {
                var directory = Path.GetDirectoryName(_path);
                if (!string.IsNullOrEmpty(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                var text = new StringBuilder();
                foreach (var claim in _claims.Values)
                {
                    text.Append(JsonConvert.SerializeObject(ToRecord(claim))).Append("\r\n");
                }

                File.WriteAllBytes(temporary, Utf8.GetBytes(text.ToString()));
                if (File.Exists(_path))
                {
                    File.Replace(temporary, _path, null);
                }
                else
                {
                    File.Move(temporary, _path);
                }

                _diskStale = false;
                return true;
            }
            catch (Exception exception)
            {
                _diskStale = true;
                _log("PDM claims: the claims file could not be written; the claims stay in memory for this process. " + exception.Message);
                TryDelete(temporary);
                return false;
            }
        }

        private static void TryDelete(string path)
        {
            try
            {
                File.Delete(path);
            }
            catch (Exception)
            {
                // A leftover temporary file is overwritten by the next write.
            }
        }

        /// <summary>The claim on one line, or null when the line is not a complete claim record.</summary>
        private static PdmClaim? TryParse(string line)
        {
            try
            {
                var record = JsonConvert.DeserializeObject<ClaimRecord>(line);
                if (record == null
                    || record.Vault == null || record.Vault.Length == 0
                    || record.FileId <= 0
                    || record.ToolId == null || record.ToolId.Length == 0
                    || record.LockedOnComputer == null
                    || record.LockPath == null
                    || !DateTime.TryParse(record.ClaimedUtc, CultureInfo.InvariantCulture,
                        DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var claimed))
                {
                    return null;
                }

                return new PdmClaim
                {
                    Vault = record.Vault,
                    FileId = record.FileId,
                    ToolId = record.ToolId,
                    LockedByUserId = record.LockedByUserId,
                    LockedOnComputer = record.LockedOnComputer,
                    LockPath = record.LockPath,
                    ClaimedUtc = claimed,
                };
            }
            catch (JsonException)
            {
                return null;
            }
        }

        private static ClaimRecord ToRecord(PdmClaim claim) => new ClaimRecord
        {
            Vault = claim.Vault,
            FileId = claim.FileId,
            ToolId = claim.ToolId,
            LockedByUserId = claim.LockedByUserId,
            LockedOnComputer = claim.LockedOnComputer,
            LockPath = claim.LockPath,
            ClaimedUtc = claim.ClaimedUtc.ToUniversalTime().ToString("o", CultureInfo.InvariantCulture),
        };

        private static PdmClaim Copy(PdmClaim claim) => new PdmClaim
        {
            Vault = claim.Vault,
            FileId = claim.FileId,
            ToolId = claim.ToolId,
            LockedByUserId = claim.LockedByUserId,
            LockedOnComputer = claim.LockedOnComputer,
            LockPath = claim.LockPath,
            ClaimedUtc = claim.ClaimedUtc,
        };

        private static string Key(string vault, int fileId) =>
            vault + "|" + fileId.ToString(CultureInfo.InvariantCulture);

        /// <summary>One line of the claims file.</summary>
        private sealed class ClaimRecord
        {
            public string? Vault { get; set; }
            public int FileId { get; set; }
            public string? ToolId { get; set; }
            public int LockedByUserId { get; set; }
            public string? LockedOnComputer { get; set; }
            public string? LockPath { get; set; }
            public string? ClaimedUtc { get; set; }
        }
    }
}
