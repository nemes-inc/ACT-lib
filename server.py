#!/usr/bin/env python3
"""
Simple HTTP server for testing the p5.js EEG ACT visualizer.
Run this script from the p5js directory to serve the files.
"""

import http.server
import socketserver
import os
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        super().end_headers()

def run_server(port=8000):
    """Run the HTTP server on the specified port.

    We serve from the project root so relative paths in JSON (e.g., 'data/...')
    are directly accessible via the browser. Open /p5js/index.html in your browser.
    """

    # Serve from project root (directory containing this server.py)
    project_root = Path(__file__).parent
    os.chdir(project_root)

    handler = CustomHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🚀 Server running at http://localhost:{port}")
            print(f"📁 Serving files from: {os.getcwd()}")
            print("📄 Open http://localhost:%d/p5js/index.html to view the visualizer" % port)
            print("🛑 Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python server.py {port + 1}")
        else:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number")
            sys.exit(1)

    run_server(port)
