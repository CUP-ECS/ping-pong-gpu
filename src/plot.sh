#!/bin/bash

#-------------------------------------------------------------------------
# Author: Keira Haskins
# Date:   07/15/2021
#
#-------------------------------------------------------------------------

#function gather_data() {
  for i in 10 20 40 80 160 320 640; do
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off ./build/ping_pong $i 1000 4 1
  done
#}

