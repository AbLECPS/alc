from http.client import HTTPConnection
import threading
import json
import base64


authorization = base64.b64encode(bytes("alc:alc", 'utf-8')).decode('utf-8')

headers = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Authorization": "Basic {0}".format(authorization)
}



def update_status_attribute(name=""):

    print("\"{0}\" starting ...".format(name))

    http_connection = HTTPConnection("localhost", port=8000, timeout=100000)

    alcmodelupdater_payload_updatestatus = {
        "projectId": "alc+ep_robustness",
        "active_node_path": "/y/h/V/Y/0",
        "message": "foobar-{0}".format(name),
        "use_lock_file": False,
        "use_merge": True
    }

    alcmodelupdater_payload_updatestatus_string = json.dumps(
        alcmodelupdater_payload_updatestatus, indent=4, sort_keys=True
    )

    http_connection.request(
        "POST", "/alcmodelupdater/updatestatus", body=alcmodelupdater_payload_updatestatus_string, headers=headers
    )

    http_response = http_connection.getresponse()

    output_headers = http_response.headers

    output_json_string = http_response.read()

    http_connection.close()

    print(output_headers)
    print(output_json_string.decode("utf-8"))



thread_1 = threading.Thread(target=update_status_attribute, kwargs={"name": "thread_1"})
thread_2 = threading.Thread(target=update_status_attribute, kwargs={"name": "thread_2"})

thread_1.start()
thread_2.start()

thread_1.join()
thread_2.join()
