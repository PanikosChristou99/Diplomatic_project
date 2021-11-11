FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y

RUN apt-get install -y python3-pip python3-dev \ 
    git \
    wget \
    wget \
    build-essential 
# We copy just the requirements.txt first to leverage Docker cache
COPY . /app

WORKDIR /app

RUN git clone https://github.com/AlexeyAB/darknet

WORKDIR /app/darknet

RUN chmod a+x /app/darknet > /dev/null 2>&1

RUN chmod +x *.sh

RUN make > /dev/null 2>&1

RUN apt install --fix-missing -y  ffmpeg libopencv-dev libgtk-3-dev python-numpy python3-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev qtbase5-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip  

RUN chmod a+x /app/darknet > /dev/null 2>&1

RUN touch /app/darknet/output.txt > /dev/null 2>&1

RUN wget https://pjreddie.com/media/files/yolov2.weights  
RUN wget https://pjreddie.com/media/files/yolov2-tiny.weights 
RUN wget https://pjreddie.com/media/files/yolov3.weights  
RUN wget https://pjreddie.com/media/files/yolov3-tiny.weights 
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights  
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights 


# RUN /app/darknet/darknet cfg/yolov2.cfg yolov2.weights /app/carsgraz_002.bmp  >> /app/darknet/output.txt \
#     /app/darknet/darknet cfg/yolov2-tiny.cfg yolov2-tiny.weights /app/carsgraz_002.bmp  >> /app/darknet/output.txt \
#     /app/darknet/darknet cfg/yolov3.cfg yolov3.weights /app/carsgraz_002.bmp  >> /app/darknet/output.txt \
#     /app/darknet/darknet cfg/yolov3-tiny.cfg yolov3-tiny.weights /app/carsgraz_002.bmp  >> /app/darknet/output.txt \
#     /app/darknet/darknet cfg/yolov4.cfg yolov4.weights /app/carsgraz_002.bmp  >> /app/darknet/output.txt \
#     /app/darknet/darknet cfg/yolov4-tiny.cfg yolov4-tiny.weights /app/carsgraz_002.bmp  >> /app/darknet/output.txt 

# RUN ./darknet detect ./cfg/yolov4-tiny.cfg ./yolov4-tiny.weights /app/carsgraz_002.bmp
