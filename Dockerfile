# cuda11.1.0, pytorch1.8 the base image is on nvcr, please login nvcr.io before using it 
# or you can build up your own pytorch env
FROM nvcr.io/nvidia/pytorch:20.11-py3

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python
RUN pip install mmcv-full
RUN pip install future tensorboard 


#################################################
# CHANGE THIS PART ACCORDINGLY !!!
ENV SHELL=/bin/bash \
    NB_USER=sguo \
    NB_UID=180691 \
    NB_GROUP=CVLAB-unit \
    NB_GID=11166
#################################################

ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}
# RUN chown $NB_USER:$NB_GID $CONDA_DIR

# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers


## user: user permission
USER $NB_USER
