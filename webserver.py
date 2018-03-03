from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import threading

def make_http_handler(output):

    class TraderHTTPHandler(BaseHTTPRequestHandler):
        def _set_response(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_GET(self):
            logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
            self._set_response()
            self.wfile.write("Crypto trader results </br></br>".encode('utf-8'))

            for line in output:
                self.wfile.write(line.format(self.path).encode('utf-8'))

        def do_POST(self):
            content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
            post_data = self.rfile.read(content_length) # <--- Gets the data itself
            logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                    str(self.path), str(self.headers), post_data.decode('utf-8'))

            self._set_response()
            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

    return TraderHTTPHandler


class TraderHTTPServer():

    def run(self, port=8080, output=[]):
        logging.basicConfig(level=logging.INFO)
        server_address = ('', port)
        handler_class=make_http_handler(output)
        httpd = HTTPServer(server_address, handler_class)


        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        try:
            logging.info('Starting httpd...\n')
            thread.start()
        except KeyboardInterrupt:
            logging.info('Stopping httpd...\n')
            httpd.shutdown()


