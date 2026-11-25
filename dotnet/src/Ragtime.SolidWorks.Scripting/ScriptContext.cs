using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

namespace Ragtime.SolidWorks.Scripting
{
    /// <summary>Services provided to a script running against one SolidWorks instance.</summary>
    public sealed class ScriptContext
    {
        private readonly Action<string> _log;

        public ScriptContext(SldWorks app, JObject arguments, string workFolder, Action<string> log)
        {
            App = app;
            Arguments = arguments;
            WorkFolder = workFolder;
            _log = log;
        }

        public SldWorks App { get; }
        public JObject Arguments { get; }
        public string WorkFolder { get; }
        public void Log(string message) => _log(message);
        public string NewWorkPath(string fileName) => ScriptWorkPath.Create(WorkFolder, fileName);

        public string? DefaultTemplate(int swDocumentTypes)
        {
            int preference;
            string extension;
            switch ((swDocumentTypes_e)swDocumentTypes)
            {
                case swDocumentTypes_e.swDocPART:
                    preference = (int)swUserPreferenceStringValue_e.swDefaultTemplatePart;
                    extension = "*.prtdot";
                    break;
                case swDocumentTypes_e.swDocASSEMBLY:
                    preference = (int)swUserPreferenceStringValue_e.swDefaultTemplateAssembly;
                    extension = "*.asmdot";
                    break;
                case swDocumentTypes_e.swDocDRAWING:
                    preference = (int)swUserPreferenceStringValue_e.swDefaultTemplateDrawing;
                    extension = "*.drwdot";
                    break;
                default:
                    return null;
            }
            var preferred = App.GetUserPreferenceStringValue(preference);
            if (!string.IsNullOrEmpty(preferred) && File.Exists(preferred)) return preferred;
            var root = ScriptTemplatePath.ForRevision(
                System.Environment.GetFolderPath(System.Environment.SpecialFolder.CommonApplicationData), App.RevisionNumber());
            if (root == null || !Directory.Exists(root)) return null;
            return Directory.GetFiles(root, extension, SearchOption.AllDirectories)
                .OrderBy(path => path, StringComparer.Ordinal)
                .FirstOrDefault();
        }
    }
}
