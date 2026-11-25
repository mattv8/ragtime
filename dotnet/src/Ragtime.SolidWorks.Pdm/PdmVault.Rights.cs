using System.Collections.Generic;
using EPDM.Interop.epdm;
using Ragtime.SolidWorks.Core;

namespace Ragtime.SolidWorks.Pdm
{
    internal sealed partial class PdmVault
    {
        public PdmFolderRights GetFolderRights(string vaultPath) => Read("read the folder rights", () =>
        {
            var folder = FindFolder(vaultPath) ?? throw PdmErrors.NotFound(vaultPath);
            return new PdmFolderRights
            {
                Read = folder.HasRightsEx((int)EdmRightFlags.EdmRight_Read, 0),
                AddFiles = folder.HasRightsEx((int)EdmRightFlags.EdmRight_Add, 0),
                CheckOut = folder.HasRightsEx((int)EdmRightFlags.EdmRight_Lock, 0),
                AddFolders = folder.HasRightsEx((int)EdmRightFlags.EdmRight_AddFolder, 0),
            };
        });

        public IReadOnlyList<string> ListSubFolderNames(string vaultPath) => Read("list the subfolders", () =>
        {
            var folder = FindFolder(vaultPath) ?? throw PdmErrors.NotFound(vaultPath);
            var names = new List<string>();
            foreach (var child in SubFolders(folder))
            {
                names.Add(child.Name);
            }

            return names;
        });
    }
}
