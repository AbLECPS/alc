import json
from http.client import HTTPConnection
import WorkflowUtils
from ProjectParameters import ProjectParameters


class UpdateStatusTask:

    logger = None

    message_key = "message"

    update_status_url_path = "/alcmodelupdater/updatestatus"

    def update_status_node(self, update_message):

        alcmodelupdater_payload = {
            WorkflowUtils.project_id_key: ProjectParameters.get_project_id(),
            WorkflowUtils.active_node_path_key: ProjectParameters.get_status_node(),
            self.message_key: update_message
        }

        http_connection = HTTPConnection(WorkflowUtils.webgme_ip_address, WorkflowUtils.webgme_port)

        alcmodelupdater_payload_string = json.dumps(alcmodelupdater_payload, indent=4, sort_keys=True)

        http_connection.request(
            "POST",
            str(self.update_status_url_path),
            body=alcmodelupdater_payload_string,
            headers=WorkflowUtils.webgme_router_header
        )

        http_response = http_connection.getresponse()

        status = http_response.status
        output_json_string = http_response.read()

        http_connection.close()

        if status == WorkflowUtils.status_ok:
            output_json = json.loads(output_json_string)
            if not output_json.get(WorkflowUtils.success_key, False):
                self.logger.warning(
                    "STATUS NODE NOT UPDATED IN WEBGME MODEL."
                )
        else:
            self.logger.warning(
                "ERROR FROM WEBGME SERVER -- COULD NOT UPDATE STATUS NODE IN WEBGME MODEL"
            )
