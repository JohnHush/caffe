#!/usr/bin/env sh
set -e

/home/jh/working_lib/caffe/build/tools/caffe train --solver=./solver.prototxt --weights=/home/jh/working_data/bvlc_reference_caffenet.caffemodel $@
