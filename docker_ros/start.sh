#!/bin/bash

CODE=${1:-`pwd`}

docker run -itd --rm \
           --ipc host \
           --network host \
           --gpus all \
           --env "NVIDIA_DRIVER_CAPABILITIES=all" \
           --env "DISPLAY=$DISPLAY" \
           --env "QT_X11_NO_MITSHM=1" \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           -v $CODE:/home/docker_user/BeyondBareQueries:rw \
           --name bbq_container-husky_demo_ros1 bbq_image-ros1
#    -p 7863:7863 \
#    -p 4433:4433 \
#    -p 4444:4444 \