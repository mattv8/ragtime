using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using SW = global::SolidWorks.Interop.sldworks;
using SWC = global::SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>Document list, silent read-only open and close, model summary and configurations.</summary>
    internal sealed partial class SolidWorksSession
    {
        private static readonly int OpenOptions =
            (int)SWC.swOpenDocOptions_e.swOpenDocOptions_Silent | (int)SWC.swOpenDocOptions_e.swOpenDocOptions_ReadOnly;

        /// <summary>All documents open in this instance, including those the host did not open.</summary>
        public IReadOnlyList<SessionDocumentInfo> ListOpenDocuments() => Run("listing the open documents", () =>
        {
            var infos = new List<SessionDocumentInfo>();
            foreach (var document in OpenDocuments())
            {
                infos.Add(Describe(document, false, null));
            }

            return infos;
        });

        /// <summary>
        /// Opens a file read-only and silently. The path must be on local storage (checked before any COM call). A
        /// document already open with that path is returned as it is (not reopened or activated). Never saves.
        /// </summary>
        public SessionDocumentInfo OpenReadOnlySilent(string path, string? configuration) => Run("opening the document", () =>
        {
            DocumentPaths.RequireOpenable(path);
            LocalFileCheck.Require(path, "path");
            var open = FindOpenDocument(new DocumentRef(path, null));
            if (open != null)
            {
                return Describe(open, true, path);
            }

            if (!File.Exists(path))
            {
                throw HostError.Coded(ErrorCodes.DocumentNotFound, "No file exists at the requested path.");
            }

            var errors = 0;
            var warnings = 0;
            var document = _application.OpenDoc6(
                path,
                DocumentTypeFromPath(path),
                OpenOptions,
                configuration ?? string.Empty,
                ref errors,
                ref warnings);
            if (document == null)
            {
                throw OpenFailure(errors, warnings);
            }

            return Describe(document, false, path);
        });

        /// <summary>Closes a document without saving. The caller only asks for documents the host opened.</summary>
        public void CloseDocument(DocumentRef document) => RunVoid("closing the document", () =>
        {
            var open = RequireOpenDocument(document);
            _application.CloseDoc(open.GetTitle());
        });

        /// <summary>Summary of a model with type-specific counts. Counts walk the feature tree once.</summary>
        public ModelSummary GetModelSummary(DocumentRef document) => Run("reading the model summary", () =>
        {
            var model = RequireOpenDocument(document);
            var type = ReaderNames.DocumentType(model.GetType());
            var features = FeatureTree.Visible(model);
            var health = features.ConvertAll(FeatureTree.Health);
            var summary = new ModelSummary
            {
                Type = type,
                Title = model.GetTitle() ?? string.Empty,
                Path = model.GetPathName() ?? string.Empty,
                ActiveConfiguration = ActiveConfigurationName(model),
                ConfigurationCount = model.GetConfigurationCount(),
                FeatureCount = features.Count,
                FeaturesWithErrors = health.FindAll(state => state.HasError).Count,
                FeaturesWithWarnings = health.FindAll(state => state.HasWarning).Count,
                Material = type == ProtocolInfo.DocumentTypes.Part ? PartMaterial(model) : null,
                CustomPropertyCount = ComHelper.Strings(model.Extension.CustomPropertyManager[string.Empty].GetNames()).Count,
                NeedsRebuild = model.Extension.NeedsRebuild,
                Units = ProtocolInfo.SiUnits,
            };

            if (type == ProtocolInfo.DocumentTypes.Part)
            {
                summary.SolidBodyCount = SolidBodyCount(model);
            }
            else if (type == ProtocolInfo.DocumentTypes.Assembly)
            {
                summary.ComponentCount = ComponentCount(model);
                summary.MateCount = MateFeatures(model).Count;
            }

            return summary;
        });

        /// <summary>All configurations of the model in SolidWorks order.</summary>
        public IReadOnlyList<ConfigurationItem> ListConfigurations(DocumentRef document) => Run("listing the configurations", () =>
        {
            var model = RequireOpenDocument(document);
            var active = ActiveConfigurationName(model);
            var names = ComHelper.Strings(model.GetConfigurationNames());
            ComWalk.RequireWithinLimit(names.Count, ComWalk.ItemLimit, "The configuration list");
            var items = new List<ConfigurationItem>();
            foreach (var name in names)
            {
                var configuration = ComHelper.TryObject(() => model.GetConfigurationByName(name) as SW.Configuration);
                items.Add(new ConfigurationItem
                {
                    Name = name,
                    IsActive = name == active,
                    Description = ComHelper.TryText(() => configuration?.Description),
                    Parent = ComHelper.TryText(() => configuration?.GetParent()?.Name),
                });
            }

            return items;
        });

        private static SessionDocumentInfo Describe(SW.ModelDoc2 document, bool wasAlreadyOpen, string? fallbackPath)
        {
            var path = document.GetPathName() ?? string.Empty;
            if (path.Length == 0)
            {
                path = fallbackPath ?? string.Empty;
            }

            return new SessionDocumentInfo(
                document.GetTitle() ?? string.Empty,
                path,
                ReaderNames.DocumentType(document.GetType()),
                ActiveConfigurationName(document),
                document.IsOpenedReadOnly(),
                document.GetSaveFlag(),
                wasAlreadyOpen);
        }

        private static int DocumentTypeFromPath(string path)
        {
            switch (Path.GetExtension(path).ToLowerInvariant())
            {
                case ".sldprt":
                    return (int)SWC.swDocumentTypes_e.swDocPART;
                case ".sldasm":
                    return (int)SWC.swDocumentTypes_e.swDocASSEMBLY;
                case ".slddrw":
                    return (int)SWC.swDocumentTypes_e.swDocDRAWING;
                default:
                    throw HostError.InvalidArguments("path", "The file must end in .sldprt, .sldasm or .slddrw.");
            }
        }

        private static HostErrorException OpenFailure(int errors, int warnings)
        {
            var details = new JObject
            {
                ["errors"] = errors,
                ["warnings"] = warnings,
            };
            if ((errors & (int)SWC.swFileLoadError_e.swFileNotFoundError) != 0)
            {
                return HostError.Coded(ErrorCodes.DocumentNotFound, "SolidWorks reports that the file was not found.", details);
            }

            return HostError.Coded(ErrorCodes.SolidworksError, "SolidWorks could not open the document read-only.", details);
        }

        /// <summary>
        /// Material assigned to the part in its active configuration. SolidWorks returns an empty name when no material is
        /// assigned, which gives null.
        /// </summary>
        private static string? PartMaterial(SW.ModelDoc2 model)
        {
            var part = ComHelper.Require(model as SW.PartDoc, "the part document");
            var configuration = ActiveConfigurationName(model) ?? string.Empty;
            var name = part.GetMaterialPropertyName2(configuration, out _);
            return string.IsNullOrWhiteSpace(name) ? null : name.Trim();
        }

        /// <summary>Number of solid bodies, including hidden ones. SolidWorks returns null when there are none.</summary>
        private static int SolidBodyCount(SW.ModelDoc2 model)
        {
            var part = ComHelper.Require(model as SW.PartDoc, "the part document");
            return ComWalk.Collection(part.GetBodies2((int)SWC.swBodyType_e.swSolidBody, false), "The solid body list").Count;
        }

        private static int ComponentCount(SW.ModelDoc2 model)
        {
            var assembly = ComHelper.Require(model as SW.AssemblyDoc, "the assembly document");
            return assembly.GetComponentCount(false);
        }
    }
}
