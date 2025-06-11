#!/bin/bash

DATA_PATH=$1
CODE=${2:-`pwd`}

if [[ ! $DATA_PATH ]]; then
    echo "Please, provide path to data"
    exit 1
fi

docker run -itd --rm \
           --ipc host \
           --network host \
           --gpus all \
           --env "NVIDIA_DRIVER_CAPABILITIES=all" \
           --env "DISPLAY=$DISPLAY" \
           --env "QT_X11_NO_MITSHM=1" \
           -p 7863:7863 \
           -p 4433:4433 \
           -p 4444:4444 \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           -v $CODE:/home/docker_user/BeyondBareQueries:rw \
           -v $DATA_PATH:/datasets/:rw \
           --name bbq_container-demo bbq_image-demo