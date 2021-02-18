import os
import glog
import logging
import urllib
import copy
import time
import msgpack_numpy as msg_np
import numpy as np
import msgpack
import traceback
from urllib.parse import urlparse
from flask import Flask
from flask import request
from flask import jsonify

import tensorflow as tf


# global

HTTP_ERROR_CODE_NO_UID = 321
HTTP_ERROR_CODE_NO_TIMETIME = 322
HTTP_ERROR_CODE_NO_MSG = 323
HTTP_ERROR_CODE_NO_UID_CHATTERS = 324
HTTP_ERROR_CODE_UNKNOWN_ACTION = 340
HTTP_ERROR_CODE_BIGAMY = 360
HTTP_ERROR_CODE_NOKEY = 401
HTTP_ERROR_CODE_NO_ACTION = 405
# flask
app = Flask(__name__)
# set flask log level
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.WARNING)

# test
tmp = {
    'a' : 'abc',
    'vec' : np.zeros((3, 3))
}
content = msg_np.packb(tmp)
content2 = msg_np.unpackb(content)
glog.info(content2['a'])
glog.info(str(content2['vec']))
############
import lstm_model
model_wrapper = lstm_model.LstmModel()
last_model_file = '1_final.h5'
glog.info('loading ' + last_model_file)
model_wrapper.load_model_from_file(file=last_model_file)

def get_timestamp_for_dingding():
    return time.strftime("[%m/%d-%H:%M:%S]", time.localtime())

def check_key(query_components):
    if not "api_key" in query_components.keys():
        glog.info("no api_key")
        return False
    if query_components["api_key"][0] != 'fdshuiafwen':
        glog.info("wrong api_key")
        return False
    return True

@app.route('/ping', methods = ['GET'])
def process_ping():
    s = "ping from " + str(request.environ['REMOTE_ADDR']) + ":" + str(request.environ['REMOTE_PORT'])
    glog.info(s)
    return s

@app.route('/get_response', methods = ['POST'])
def process_get_response():
    glog.info('get_response is called from ' + str(request.environ['REMOTE_ADDR']) + ":" + str(request.environ['REMOTE_PORT']))
    query = urlparse(request.url).query
    if not query:
        # this might be attacker
        return "hello world"
    query_components = urllib.parse.parse_qs(query)
    # https://stackoverflow.com/questions/8928730/processing-http-get-input-parameter-on-server-side-in-python
    if not check_key(query_components):
        return "no key", HTTP_ERROR_CODE_NOKEY
    decode_content = msg_np.unpackb(request.data)
    session_id = decode_content['session_id']
    vec_in = decode_content['vec_in']
    glog.info('predict vec_in shape ' + str(vec_in.shape))
    vec_out = model_wrapper.predict(vec_in)
    ret_data = {
        "session_id" : session_id,
        "vec_out" : vec_out
    }
    response = app.response_class(
        response=msg_np.packb(ret_data),
        status=200,
        mimetype='application/msgpack_numpy'
    )
    return response
