#!/bin/bash

source /Users/peizanwang/Programs/tensor/bin/activate

spark-submit --master local\[4\] soy.py