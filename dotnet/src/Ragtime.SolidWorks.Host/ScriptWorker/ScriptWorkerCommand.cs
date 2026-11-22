using System;
using System.CodeDom.Compiler;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.CSharp;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Ragtime.SolidWorks.Core;
using Ragtime.SolidWorks.Runtime;
using Ragtime.SolidWorks.Scripting;
using SolidWorks.Interop.sldworks;

namespace Ragtime.SolidWorks.Host
{
    internal static class ScriptWorkerCommand
    {
        public static int Run(string folder)
        {
            try
            {
                var request = ScriptWorkerFiles.ReadRequest(folder);
                var result = CompileAndRun(folder, request);
                ScriptWorkerFiles.WriteResult(folder, result);
                return 0;
            }
            catch (Exception error)
            {
                try { ScriptWorkerFiles.WriteResult(folder, Failure(error)); }
                catch (Exception) { }
                return 2;
            }
        }

        private static JObject CompileAndRun(string folder, JObject request)
        {
            var watch = Stopwatch.StartNew();
            var output = Path.Combine(folder, "script.dll");
            var temp = Path.Combine(folder, "compile");
            Directory.CreateDirectory(temp);
            var source = File.ReadAllText(Path.Combine(folder, ScriptWorkerFiles.SourceName), Encoding.UTF8);
            var host = (string?)request["host_folder"] ?? throw new InvalidDataException("Missing host_folder.");
            CompilerResults compiled;
            try
            {
                using (var provider = new CSharpCodeProvider())
                {
                    var options = new CompilerParameters
                    {
                        GenerateExecutable = false,
                        GenerateInMemory = false,
                        OutputAssembly = output,
                        TempFiles = new TempFileCollection(temp, true),
                        CompilerOptions = "/langversion:5 /platform:x64",
                    };
                    foreach (var name in new[] { "System.dll", "System.Core.dll", "Microsoft.CSharp.dll" })
                        options.ReferencedAssemblies.Add(name);
                    foreach (var name in new[] { "SolidWorks.Interop.sldworks.dll", "SolidWorks.Interop.swconst.dll", "Newtonsoft.Json.dll", "Ragtime.SolidWorks.Scripting.dll" })
                        options.ReferencedAssemblies.Add(Path.Combine(host, name));
                    compiled = provider.CompileAssemblyFromSource(options, source);
                }
            }
            catch (Exception)
            {
                return new JObject { ["status"] = "compile_unavailable", ["compile_ms"] = watch.ElapsedMilliseconds };
            }
            var compileMs = watch.ElapsedMilliseconds;
            if (compiled.Errors.HasErrors)
            {
                var errors = new JArray();
                foreach (CompilerError error in compiled.Errors)
                {
                    if (error.IsWarning) continue;
                    errors.Add(new JObject
                    {
                        ["line"] = error.Line, ["column"] = error.Column, ["code"] = error.ErrorNumber,
                        ["message"] = error.ErrorText, ["hint"] = ScriptCompileHints.HintFor(error.ErrorNumber, error.ErrorText),
                    });
                }
                return new JObject { ["status"] = "compile_failed", ["compile_errors"] = errors, ["compile_ms"] = compileMs };
            }
            if (compiled.NativeCompilerReturnValue != 0 || !File.Exists(output))
                return new JObject { ["status"] = "compile_unavailable", ["compile_ms"] = compileMs };

            var type = Assembly.LoadFrom(output).GetType("RagtimeScript", false);
            var method = type?.GetMethod("Run", BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(ScriptContext) }, null);
            if (type == null || !type.IsPublic || !type.IsAbstract || !type.IsSealed || method == null || method.ReturnType != typeof(object))
                return new JObject { ["status"] = "invalid", ["compile_ms"] = compileMs };

            JObject? result = null;
            Exception? failure = null;
            var thread = new Thread(() =>
            {
                try { result = Execute(folder, request, method); }
                catch (Exception error) { failure = error; }
            });
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();
            thread.Join();
            if (failure != null) throw failure;
            result!["compile_ms"] = compileMs;
            return result;
        }

        private static JObject Execute(string folder, JObject request, MethodInfo method)
        {
            var deadline = (DateTime?)request["deadline_utc"] ?? throw new InvalidDataException("Missing deadline.");
            var filter = new ScriptMessageFilter(deadline.ToUniversalTime());
            ScriptMessageFilter.Register(filter, out var previous);
            try
            {
                SldWorks? app;
                try
                {
                    var pid = (int?)request["process_id"] ?? 0;
                    using (var process = Process.GetProcessById(pid))
                    {
                        var expected = (DateTime?)request["process_start_utc"] ?? DateTime.MinValue;
                        if (process.StartTime.ToUniversalTime() != expected.ToUniversalTime())
                            return new JObject { ["status"] = "attach_failed" };
                    }
                    app = RunningObjectTable.FindApplication(pid) as SldWorks;
                    if (app == null || app.GetProcessID() != pid) return new JObject { ["status"] = "attach_failed" };
                }
                catch (Exception) { return new JObject { ["status"] = "attach_failed" }; }

                var attached = (bool?)request["attached"] ?? false;
                var oldUserControl = attached && app.UserControl;
                var oldVisible = attached && app.Visible;
                var visibility = new bool[3];
                var captured = 0;
                var logPath = Path.Combine(folder, ScriptWorkerFiles.LogName);
                using (var file = new StreamWriter(logPath, true, new UTF8Encoding(false)) { AutoFlush = true })
                {
                    var log = new ScriptLogBuffer(line => file.WriteLine(JsonConvert.SerializeObject(line)));
                    try
                    {
                        if (!attached) { app.UserControl = false; app.Visible = false; }
                        for (var i = 0; i < visibility.Length; i++)
                        {
                            visibility[i] = app.GetDocumentVisible(i + 1);
                            captured++;
                            app.DocumentVisible(true, i + 1);
                        }
                        var context = new ScriptContext(app, (JObject?)request["arguments"] ?? new JObject(),
                            (string?)request["work_folder"] ?? throw new InvalidDataException("Missing work_folder."), log.Log);
                        var watch = Stopwatch.StartNew();
                        try
                        {
                            var value = method.Invoke(null, new object[] { context });
                            watch.Stop();
                            return new JObject { ["status"] = "completed", ["result"] = ScriptResultGuard.Serialize(value, Marshal.IsComObject), ["run_ms"] = watch.ElapsedMilliseconds, ["log_truncated"] = log.Truncated };
                        }
                        catch (Exception error)
                        {
                            watch.Stop();
                            var actual = error is TargetInvocationException invocation && invocation.InnerException != null ? invocation.InnerException : error;
                            var result = Failure(actual);
                            result["run_ms"] = watch.ElapsedMilliseconds;
                            result["log_truncated"] = log.Truncated;
                            return result;
                        }
                    }
                    finally
                    {
                        Exception? restoreFailure = null;
                        for (var i = captured - 1; i >= 0; i--)
                        {
                            try { app.DocumentVisible(visibility[i], i + 1); }
                            catch (Exception error) { restoreFailure = restoreFailure ?? error; }
                        }
                        if (attached)
                        {
                            try { app.UserControl = oldUserControl; }
                            catch (Exception error) { restoreFailure = restoreFailure ?? error; }
                            try { app.Visible = oldVisible; }
                            catch (Exception error) { restoreFailure = restoreFailure ?? error; }
                        }
                        if (restoreFailure != null) throw restoreFailure;
                    }
                }
            }
            finally { ScriptMessageFilter.Register(previous, out _); }
        }

        private static JObject Failure(Exception error) => new JObject
        {
            ["status"] = "failed",
            ["exception_type"] = error.GetType().Name,
            ["message"] = error.Message,
            ["stack"] = string.Join("\n", (error.StackTrace ?? string.Empty).Split('\n').Where(line => line.IndexOf("RagtimeScript", StringComparison.Ordinal) >= 0)),
        };
    }
}
