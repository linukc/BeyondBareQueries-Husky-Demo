from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl
import io
import zipfile
from multiprocessing import Process

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        request_data = self.rfile.read(content_length).decode('utf-8')
        print("Received request:", request_data)

        try:
            with open('depth.png', 'rb') as img1, open('image.png', 'rb') as img2:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as zipf:
                    zipf.writestr('depth.png', img1.read())
                    zipf.writestr('image.png', img2.read())
                buffer.seek(0)

                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", "attachment; filename=images.zip")
                self.end_headers()
                self.wfile.write(buffer.read())

        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"One or both images not found.")

def run_server():
    server_address = ('0.0.0.0', 4433)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

    # Modern SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile='server.pem')  # Use PEM file with cert + key

    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print("Server running on https://localhost:4433")
    httpd.serve_forever()

if __name__ == '__main__':
    server_process = Process(target=run_server)
    server_process.start()

    print("Main process continues...")
    input("Press Enter to stop the server...\n")
    server_process.terminate()
    server_process.join()
    print("Server stopped.")
