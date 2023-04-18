from http.client import HTTPConnection
import json
import base64


http_connection_createdatanode_1 = HTTPConnection("localhost", port=8000, timeout=100000)
http_connection_createdatanode_2 = HTTPConnection("localhost", port=8000, timeout=100000)
# http_connection_updatestatus = HTTPConnection("localhost", port=8000)


authorization = base64.b64encode(bytes("alc:alc", 'utf-8')).decode('utf-8')

headers = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Authorization": "Basic {0}".format(authorization)
}

alcmodelupdater_payload_createdatanode = {
    "projectId": "alc+ep_robustness",
    "active_node_path": "/V",
    "name": "foo",
    "modifications": {}
}

# alcmodelupdater_payload_updatestatus = {
#     "projectId": "alc+ep_robustness",
#     "active_node_path": "/y/h/n/q/F",
#     "message": "foobar"
# }

alcmodelupdater_payload_createdatanode_string = json.dumps(
    alcmodelupdater_payload_createdatanode, indent=4, sort_keys=True
)

# alcmodelupdater_payload_updatestatus_string = json.dumps(
#     alcmodelupdater_payload_updatestatus, indent=4, sort_keys=True
# )


http_connection_createdatanode_1.request(
    "POST", "/alcmodelupdater/createdatanode", body=alcmodelupdater_payload_createdatanode_string, headers=headers
)
http_connection_createdatanode_2.request(
    "POST", "/alcmodelupdater/createdatanode", body=alcmodelupdater_payload_createdatanode_string, headers=headers
)
# http_connection_updatestatus.request(
#     "POST", "/alcmodelupdater/updatestatus", body=alcmodelupdater_payload_updatestatus_string, headers=headers
# )


http_response_createdatanode_1 = http_connection_createdatanode_1.getresponse()
http_response_createdatanode_2 = http_connection_createdatanode_2.getresponse()
# http_response_updatestatus = http_connection_updatestatus.getresponse()

status_createdatanode_1 = http_response_createdatanode_1.status
status_createdatanode_2 = http_response_createdatanode_2.status
# status_updatestatus = http_response_updatestatus.status

output_headers_createdatanode_1 = http_response_createdatanode_1.headers
output_headers_createdatanode_2 = http_response_createdatanode_2.headers
# output_headers_status = http_response_updatestatus.headers

output_json_string_createdatanode_1 = http_response_createdatanode_1.read()
output_json_string_createdatanode_2 = http_response_createdatanode_2.read()
# output_json_string_status = http_response_updatestatus.read()

http_connection_createdatanode_1.close()
http_connection_createdatanode_2.close()
# http_connection_updatestatus.close()

print(status_createdatanode_1)
print(output_headers_createdatanode_1)
print(output_json_string_createdatanode_1.decode("utf-8"))
print()

print(status_createdatanode_2)
print(output_headers_createdatanode_2)
print(output_json_string_createdatanode_2.decode("utf-8"))
print()

# print(status_updatestatus)
# print(output_headers_status)
# print(output_json_string_status.decode("utf-8"))
# print()
