FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

SHELL ["/bin/bash", "-c"]
WORKDIR /
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update &&\
 DEBIAN_FRONTEND="noninteractive" apt -y install git tree vim wget openssh-client curl libsm6 libxext6 ffmpeg && \
 apt clean && \
 rm -rf /var/lib/apt/lists/* && \
 rm -rf /tmp/*


RUN rm /usr/local/bin/pip && \
 rm /usr/local/bin/pip3 && \
 apt update && apt install python3-pip --reinstall -y && \
 pip3 install pytype && \
 pip3 install lxml && \
 pip3 install pillow && \
 pip3 install psutil && \
 pip3 install opencv-python && \
 pip3 install GitPython && \
 pip3 install deepdiff && \
 pip3 install tfa-nightly && \
 pip3 install pandas && \
 pip3 install Shapely && \
 pip3 install cython && \
 pip3 install matplotlib && \
 pip3 install scipy && \
 pip3 install opencv_contrib_python && \
 pip3 install pandas && \
 pip3 install numpy && \
 pip3 install absl_py 
 #pip3 install absl 
 #pip3 install torch==1.6.0+cu101

ADD deep_sort deep_sort
ADD deep_sort_world_coord deep_sort_world_coord
ADD deep_sort_new_assignment deep_sort_new_assignment
ADD trackeval_world_coord trackeval_world_coord
ADD tools tools
ADD scripts scripts
ADD trackeval trackeval
ADD cosine_metric_learning cosine_metric_learning 
ADD Crop_bbox_for_train.py  Crop_bbox_for_train.py 
ADD csv_reader.py csv_reader.py
ADD tracker_app.py tracker_app.py

ADD tracker_app_world_coord.py tracker_app_world_coord.py
ADD filter_gt_track_csv.py filter_gt_track_csv.py
ADD run_eval.py run_eval.py
RUN rm -R workspace

CMD ["-c"]


