import os
import json
import time
import base64
from http.client import HTTPConnection

username = "alc"
password = "alc"
user_arg = "{0}:{1}".format(username, password)

webgme_ip_address = "127.0.0.1"
webgme_host = "localhost"
webgme_port = 8888
webgme_url = "http://{0}:{1}".format(webgme_host, webgme_port)


json_content_type_string = "application/json"
url_encoded_content_type_string = "application/x-www-form-urlencoded"

authorization = "{0}:{1}".format(username, password)
authorization_base64 = base64.b64encode(
    bytes(authorization, "utf-8")).decode("utf-8")
authorization_header_value = "Basic {0}".format(authorization_base64)

project_id_key = "projectId"
active_node_path_key = "active_node_path"
modification_key = "modifications"
set_key = "sets"
url_key = "url"
position_key = "position"
success_key = "success"
name_key = "name"
node_path_key = "node_path"

create_data_node_url_path = "/alcmodelupdater/createdatanode"
create_jupyter_node_url_path = "/alcmodelupdater/createjupyternode"
status_ok = 200
status_error = 500


webgme_router_header = {
    "Accept": "*/*",
    "Content-Type": json_content_type_string,
    "Authorization": authorization_header_value
}


def create_data_node(logger, project_owner, project_name, active_node_path, execution_name, modification={}, set_data={}):

    data_node = None

    alcmodelupdater_payload = {
        project_id_key: "{0}+{1}".format(project_owner, project_name),
        active_node_path_key: active_node_path,
        name_key: execution_name,
        modification_key: modification,
        set_key: set_data

    }

    http_connection = HTTPConnection(webgme_ip_address, webgme_port)

    alcmodelupdater_payload_string = json.dumps(
        alcmodelupdater_payload, indent=4, sort_keys=True)

    http_connection.request(
        "POST",
        str(create_data_node_url_path),
        body=alcmodelupdater_payload_string,
        headers=webgme_router_header
    )

    http_response = http_connection.getresponse()

    status = http_response.status
    output_json_string = http_response.read()

    http_connection.close()

    if status == status_ok:
        output_json = json.loads(output_json_string)
        if node_path_key in output_json:
            data_node = output_json.get(node_path_key)
        else:
            logger.warning(
                "DATA NODE NOT CREATED IN WEBGME MODEL.  Will not be able to show status of slurm"
                " job or present results."
            )
    else:
        logger.warning(
            "ERROR FROM WEBGME SERVER -- COULD NOT CREATE DATA NODE IN WEBGME MODEL.   Will not be able "
            "to show status of slurm job or present results."
        )

    return data_node


def create_jupyter_node(logger, project_owner, project_name, active_node_path, execution_name, url, position):

    jupyter_node = None

    alcmodelupdater_payload = {
        project_id_key: "{0}+{1}".format(project_owner, project_name),
        active_node_path_key: active_node_path,
        name_key: execution_name,
        url_key: url,
        position_key: position

    }

    http_connection = HTTPConnection(webgme_ip_address, webgme_port)

    alcmodelupdater_payload_string = json.dumps(
        alcmodelupdater_payload, indent=4, sort_keys=True)

    http_connection.request(
        "POST",
        str(create_jupyter_node_url_path),
        body=alcmodelupdater_payload_string,
        headers=webgme_router_header
    )

    http_response = http_connection.getresponse()

    status = http_response.status
    output_json_string = http_response.read()

    http_connection.close()

    if status == status_ok:
        output_json = json.loads(output_json_string)
        if node_path_key in output_json:
            jupyter_node = output_json.get(node_path_key)
        else:
            logger.warning(
                "JUPYTER NODE NOT CREATED IN WEBGME MODEL."
            )
    else:
        logger.warning(
            "ERROR FROM WEBGME SERVER -- COULD NOT CREATE JUPYTER NODE IN WEBGME MODEL."
        )

    return jupyter_node
