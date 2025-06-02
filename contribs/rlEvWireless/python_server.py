from http.server import HTTPServer, BaseHTTPRequestHandler
from rlevmatsim.envs.ocp.

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Hello, world!')

httpd = HTTPServer(('localhost', 8080), SimpleHandler)
httpd.serve_forever()




class SimLearner:
    def __init__(self, 
                 charge_model: 
                 ):
        self.charge_model = charge_model