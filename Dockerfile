# https://www.tensorflow.org/install/source#tested_build_configurations
FROM mirror.ccs.tencentyun.com/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
RUN echo "deb http://mirrors.cloud.tencent.com/ubuntu/ bionic main restricted universe multiverse" > /etc/apt/sources.list
RUN echo "deb http://mirrors.cloud.tencent.com/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb http://mirrors.cloud.tencent.com/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.cloud.tencent.com/ubuntu/ bionic main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.cloud.tencent.com/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.cloud.tencent.com/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install wget -y
RUN wget -O /etc/apt/sources.list http://mirrors.cloud.tencent.com/repo/ubuntu18_sources.list
RUN apt-get update
RUN apt-get install vim python3-dev python3-pip -y
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/pip3 config --user set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
RUN /usr/bin/pip3 config --user set global.trusted-mirrors.cloud.tencent.com tcloud
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip3 install Pandas matplotlib sklearn glog msgpack_numpy flask
# https://www.tensorflow.org/install/source#tested_build_configurations
# tf 2.3 works with cudnn7 and cuda10.1
RUN pip3 install tensorflow==2.3 keras
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-10.1/targets/x86_64-linux/lib/:/usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs/