const http = require("http");
http.createServer((request, response) => {
  if (request.url === "/asset.js") {
    response.writeHead(200, { "Content-Type": "application/javascript" });
    return response.end("window.runtimeBenchmarkAssetLoaded = true;");
  }
  response.writeHead(200, { "Content-Type": "text/html" });
  response.end("<main id=\"runtime-benchmark-ready\">node fixture ready</main><script src=\"/asset.js\"></script>");
}).listen(process.env.PORT || 5173, "127.0.0.1");
