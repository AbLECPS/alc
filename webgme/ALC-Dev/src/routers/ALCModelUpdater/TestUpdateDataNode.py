from http.client import HTTPConnection
import json
import base64


http_connection = HTTPConnection("localhost", port=8000)

authorization = base64.b64encode(bytes("alc:alc", 'utf-8')).decode('utf-8')

headers = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Authorization": "Basic {0}".format(authorization)
}

alcmodelupdater_payload = {
    "projectId": "alc+ep_robustness",
    "active_node_path": "/V/Eu/F",
    "modifications": {
        "datainfo": "thisnthat",
        "jobstatus": "Finished_w_Errors",
        "resultDir": "this/that/other"
    }
}

alcmodelupdater_payload_string = json.dumps(alcmodelupdater_payload, indent=4, sort_keys=True)

http_connection.request(
    "POST", "/alcmodelupdater/updatedatanode", body=alcmodelupdater_payload_string, headers=headers
)

http_response = http_connection.getresponse()

status = http_response.status
headers = http_response.headers
output_json_string = http_response.read()

http_connection.close()

print(status)
print(headers)
print(output_json_string.decode("utf-8"))
