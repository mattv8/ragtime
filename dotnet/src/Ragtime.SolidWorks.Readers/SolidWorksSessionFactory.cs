using System;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>Creates model-reading sessions for a running SolidWorks application (typed as the interop ISldWorks).</summary>
    public sealed class SolidWorksSessionFactory : ISolidWorksSessionFactory
    {
        /// <summary>Creates the factory.</summary>
        public SolidWorksSessionFactory()
        {
        }

        /// <summary>Wraps the SolidWorks application object in a read-only session. Called on the instance's STA thread.</summary>
        public ISolidWorksSession Create(object solidworksApplication, SessionOptions options)
        {
            var application = solidworksApplication as SW.ISldWorks;
            if (application == null)
            {
                throw HostError.Coded(ErrorCodes.SolidworksError, "The object is not a SolidWorks application and cannot be read.");
            }

            return new SolidWorksSession(application, options ?? throw new ArgumentNullException(nameof(options)));
        }
    }
}
