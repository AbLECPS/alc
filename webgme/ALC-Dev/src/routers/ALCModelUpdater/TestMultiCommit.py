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


lock = threading.Lock()
lock.acquire()


def create_data_node(name="", useLock=False):

    print("\"{0}\" starting ...".format(name))
    if useLock:
        print("\"{0}\" waiting for lock.".format(name))
        lock.acquire()
        print("\"{0}\" lock acquired.".format(name))

    http_connection = HTTPConnection("localhost", port=8000, timeout=100000)

    alcmodelupdater_payload_createdatanode = {
        "projectId": "alc+ep_robustness",
        "active_node_path": "/V",
        "name": "foo-{0}".format(name),
        "modifications": {},
        "use_lock_file": False,
        "use_merge": True
    }

    alcmodelupdater_payload_createdatanode_string = json.dumps(
        alcmodelupdater_payload_createdatanode, indent=4, sort_keys=True
    )

    http_connection.request(
        "POST", "/alcmodelupdater/createdatanode", body=alcmodelupdater_payload_createdatanode_string, headers=headers
    )

    http_response = http_connection.getresponse()

    output_headers = http_response.headers

    output_json_string = http_response.read()

    http_connection.close()

    print(output_headers)
    print(output_json_string.decode("utf-8"))

    if lock.locked():
        print("\"{0}\" releasing lock.".format(name))
        lock.release()
        print("\"{0}\" lock released.".format(name))


thread_1 = threading.Thread(target=create_data_node, kwargs={"name": "thread_1"})
thread_2 = threading.Thread(target=create_data_node, kwargs={"name": "thread_2"})
thread_3 = threading.Thread(target=create_data_node, kwargs={"name": "thread_3", "useLock": True})

thread_1.start()
thread_2.start()
thread_3.start()

thread_1.join()
thread_2.join()
thread_3.join()
