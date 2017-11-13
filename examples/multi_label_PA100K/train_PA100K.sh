#!/usr/bin/env sh
set -e

/Users/pitaloveu/WORKING_LIB/ssd_caffe/build/tools/caffe train --solver=/Users/pitaloveu/WORKING_LIB/ssd_caffe/examples/multi_label_PA100K/solver.prototxt --weights=/Users/pitaloveu/WORKING_LIB/ssd_caffe/examples/multi_label_PA100K/bvlc_reference_caffenet.caffemodel $@
