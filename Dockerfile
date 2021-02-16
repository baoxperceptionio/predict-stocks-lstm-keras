FROM mirror.ccs.tencentyun.com/tensorflow/tensorflow:latest-gpu-jupyter
RUN pip3 config --user set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
RUN pip3 config --user set global.trusted-mirrors.cloud.tencent.com tcloud
RUN apt-get install wget
RUN wget -O /etc/apt/sources.list http://mirrors.cloud.tencent.com/repo/ubuntu18_sources.list
RUN apt-get update
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip3 install Pandas Keras matplotlib sklearn glog msgpack_numpy flask