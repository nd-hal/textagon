#!/usr/bin/python3

import os
#print(os.getpid())
f = open("pid.txt", "w")
f.write(str(os.getpid()))
f.close()

import http.server, socketserver, socket

ip = '0.0.0.0'
port = 7777

server = http.server.HTTPServer((ip, port), http.server.CGIHTTPRequestHandler)

print("Serving HTTP on ", ip, " port ", port, " (http://", ip, ":", port, ") ...", sep = "")
server.serve_forever()
