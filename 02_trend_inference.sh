#!/bin/bash

${SPARK_HOME}/bin/spark-submit --master ${MASTER} --conf spark.cores.max=8 --conf spark.task.cpus=4 ./NXP_inference.py
