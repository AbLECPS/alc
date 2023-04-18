"""
This is a simple communication module.
It contains the functions that is necessary for two way communication.
This module can be used as-is or as a template if details of the 
communication pattern needs to be changed.
"""

import json
import random
import string
import time
from os.path import exists
from datetime import datetime

communication_file = None

# needs to be set at the beggining of execution so we know what file to look for
def set_communication_file (path):
    global communication_file
    communication_file = path
    

# our main message, it is going to constantly read the file content until 
# there is the content identified with the expected message exchange id
# the id will be generated when a message is sent to the user
# this means that only sequential messaging is allowed 
def wait_for_message (m_id):
    got_message = False
    content = None

    while not got_message:
        try:
            if exists(communication_file):
                f = open(communication_file)
                message = json.loads(f.read())
                if 'id' in message and message['id'] == m_id:
                    content = message['content']
                    got_message = True
                f.close()
        except:
            content = None
            got_message = True
        time.sleep(1)
    
    return content

# simple random id generator to identify message exchanges
def generate_exchange_id ():
    timestamp = str((datetime.now()-datetime(1,1,1)).total_seconds())
    m_id = 'M - ' + timestamp + ' - '
    m_id += ''.join(random.choice(string.digits) for i in range(10))

    return m_id

# combined function that sends a message and waits for the response
def ask_user (plugin, content):
    message = {}
    m_id = generate_exchange_id()
    message['id'] = m_id
    message['message'] = 'User question being asked...'
    message['content'] = content
    message['severity'] = 'warn'

    plugin.send_notification(message)
    response = wait_for_message(m_id)
    
    return response


