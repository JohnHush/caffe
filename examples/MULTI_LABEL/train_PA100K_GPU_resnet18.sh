#!/usr/bin/env sh
set -e

/home/jh/working_lib/caffe/build/tools/caffe train --solver=./solver_GPU_resnet18.prototxt --weights=/home/jh/working_data/models/ResNet-18-Caffemodel-on-ImageNet/resnet-18.caffemodel -gpu=0,1
