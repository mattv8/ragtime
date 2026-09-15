import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

db_path = Path(".ragtime/db/app.sqlite3")
db_path.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(db_path) as connection:
    connection.execute("create table if not exists benchmark_events (id integer primary key, name text)")
    connection.execute("insert into benchmark_events (name) values ('ready')")
    connection.commit()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/asset.js":
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.end_headers()
            self.wfile.write(b"window.runtimeBenchmarkAssetLoaded = true;")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b'<main id="runtime-benchmark-ready">python sqlite fixture ready</main><script src="/asset.js"></script>')

    def log_message(self, _format, *_args):
        pass


HTTPServer(("127.0.0.1", int(os.environ.get("PORT", "5173"))), Handler).serve_forever()
