#!/bin/bash

#-------------------------------------------------------------------------
# Author: Keira Haskins
# Date:   07/15/2021
#
#-------------------------------------------------------------------------

#function gather_data() {
  #for i in 10 20 40 80 160 320 640; do
  for i in 320 640; do
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 0 0
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 1 0
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 2 0
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 0 1
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 1 1
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 2 1
    #lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 0 2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 1 2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 2 2
  done
#}

