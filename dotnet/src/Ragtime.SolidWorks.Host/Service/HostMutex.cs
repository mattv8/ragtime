using System;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Threading;

namespace Ragtime.SolidWorks.Host
{
    /// <summary>The per-user single-instance mutex is not ours and cannot be verified; serve must not run as if it were.</summary>
    internal sealed class HostMutexException : Exception
    {
        public HostMutexException(string message)
            : base(message)
        {
        }
    }

    /// <summary>The per-user single-instance mutex (Global\ namespace so every session sees it).</summary>
    internal sealed class HostMutex : IDisposable
    {
        private readonly Mutex _mutex;

        private HostMutex(Mutex mutex)
        {
            _mutex = mutex;
        }

        /// <summary>
        /// Takes the mutex. Returns null when another serve of this user already holds it. Throws
        /// <see cref="HostMutexException"/> when the name exists but cannot be opened or was not created by this account.
        /// </summary>
        public static HostMutex? TryAcquire(string name, string sid)
        {
            Mutex mutex;
            bool createdNew;
            try
            {
                mutex = new Mutex(true, name, out createdNew, OwnerOnly(sid));
            }
            catch (UnauthorizedAccessException exception)
            {
                throw new HostMutexException("the mutex already exists and this account cannot open it (" + exception.Message + ")");
            }

            if (createdNew)
            {
                return new HostMutex(mutex);
            }

            var owned = IsOwnedBy(mutex, sid);
            mutex.Dispose();
            if (!owned)
            {
                throw new HostMutexException("the mutex already exists and was not created by this account");
            }

            return null;
        }

        /// <summary>True while some host process holds the mutex. Throws when the mutex exists but cannot be opened.</summary>
        public static bool IsHeld(string name)
        {
            try
            {
                using (Mutex.OpenExisting(name))
                {
                    return true;
                }
            }
            catch (WaitHandleCannotBeOpenedException)
            {
                return false;
            }
        }

        /// <inheritdoc />
        public void Dispose()
        {
            try
            {
                _mutex.ReleaseMutex();
            }
            catch (ApplicationException)
            {
                // Not owned by this thread any more.
            }

            _mutex.Dispose();
        }

        /// <summary>Owner and only ACE: the account's SID with FullControl.</summary>
        private static MutexSecurity OwnerOnly(string sid)
        {
            var identity = new SecurityIdentifier(sid);
            var security = new MutexSecurity();
            security.SetOwner(identity);
            security.AddAccessRule(new MutexAccessRule(identity, MutexRights.FullControl, AccessControlType.Allow));
            return security;
        }

        /// <summary>True when the existing mutex's owner is the account. Any failure to read it counts as not owned.</summary>
        private static bool IsOwnedBy(Mutex mutex, string sid)
        {
            try
            {
                var owner = mutex.GetAccessControl().GetOwner(typeof(SecurityIdentifier)) as SecurityIdentifier;
                return owner != null && string.Equals(owner.Value, sid, StringComparison.OrdinalIgnoreCase);
            }
            catch (Exception)
            {
                return false;
            }
        }
    }
}
