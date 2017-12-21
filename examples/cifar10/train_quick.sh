#!/usr/bin/env sh
set -e

TOOLS=./build/tools
LOG=./log/train-`date +%Y-%m-%d-%H-%M-%S`.log

$TOOLS/caffe train --gpu 0 \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt  2>&1 | tee $LOG

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train  --gpu 0 \
  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5  2>&1 | tee $LOG
