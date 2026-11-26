using System;
using System.Collections.Generic;
using SW = global::SolidWorks.Interop.sldworks;

namespace Ragtime.SolidWorks.Readers
{
    /// <summary>
    /// SolidWorks' explanation of feature errors, from <c>ModelDocExtension.GetWhatsWrong</c>, read once per call and looked up
    /// by feature name. The same source as <c>feature.get</c>. An unavailable table is empty, so every lookup gives "".
    /// </summary>
    internal sealed class WhatsWrongTable
    {
        private readonly Dictionary<string, string> _messages;

        private WhatsWrongTable(Dictionary<string, string> messages)
        {
            _messages = messages;
        }

        /// <summary>Reads the table of a model. Never throws.</summary>
        public static WhatsWrongTable Read(SW.ModelDoc2 model)
        {
            var messages = new Dictionary<string, string>(StringComparer.Ordinal);
            try
            {
                model.Extension.GetWhatsWrong(out object names, out object codes, out object texts);
                var featureNames = ComHelper.Strings(names);
                var featureTexts = ComHelper.Strings(texts);
                for (var index = 0; index < featureNames.Count && index < featureTexts.Count; index++)
                {
                    if (!messages.ContainsKey(featureNames[index]))
                    {
                        messages[featureNames[index]] = featureTexts[index];
                    }
                }
            }
            catch (Exception)
            {
                messages.Clear();
            }

            return new WhatsWrongTable(messages);
        }

        /// <summary>The explanation for a feature (the first match by name), or "" when there is none.</summary>
        public string For(string featureName) => _messages.TryGetValue(featureName, out var text) ? text : string.Empty;
    }
}
