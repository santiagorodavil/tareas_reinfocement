FROM charines/keras-rl-notebook:latest 
SHELL ["/bin/bash", "-c"]
USER root
RUN apt-get update \
    && apt-get install -y python-opengl nvidia-340 mesa-utils
USER jovyan
