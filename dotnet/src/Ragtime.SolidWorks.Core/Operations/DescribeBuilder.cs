using System.Linq;

namespace Ragtime.SolidWorks.Core
{
    /// <summary>Generates the <c>describe</c> result from the operation registry. Internal operations are never listed.</summary>
    public static class DescribeBuilder
    {
        /// <summary>Builds the host contract for the given registry and host version.</summary>
        public static DescribeResult Build(OperationRegistry registry, string hostVersion) =>
            new DescribeResult
            {
                Protocol = ProtocolInfo.Version,
                HostVersion = hostVersion,
                Operations = registry.Public
                    .Select(operation => new OperationDescriptor
                    {
                        Name = operation.Name,
                        Summary = operation.Summary,
                        Access = operation.Access,
                        Paged = operation.Paged,
                        Arguments = operation.ArgumentsSchema,
                        Result = operation.Result,
                    })
                    .ToList(),
            };
    }
}
