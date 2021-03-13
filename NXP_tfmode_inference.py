# Copyright 2018 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# This example demonstrates how to leverage Spark for parallel inferencing from a SavedModel.
#
# Normally, you can use TensorFlowOnSpark to just form a TensorFlow cluster for training and inferencing.
# However, in some situations, you may have a SavedModel without the original code for defining the inferencing
# graph.  In these situations, we can use Spark to instantiate a single-node TensorFlow instance on each executor,
# where each executor can independently load the model and inference on input data.
#
# Note: this particular example demonstrates use of `tf.data.Dataset` to read the input data for inferencing,
# but it could also be adapted to just use an RDD of TFRecords from Spark.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf


model_path = "/usr/local/TensorFlowOnSpark/Zabbix_24hr_prediction/TFoS_24hr_Server_failure_prediction/Zabbix"
ATKH_mem_used_csv_path = "hdfs://master:9000/user/data/ATKH_Oplus_TWGKHHPSK1MSB04_memory_usage_2020_10.csv"

def inference(args, ctx):

  # load saved_model
  saved_model = tf.saved_model.load(args.export_dir, tags='serve')
  predict = saved_model.signatures['serving_default']
  
  
  

  # define a new tf.data.Dataset (for inferencing)
  tfds = tf.data.TFRecordDataset(ATKH_mem_used_csv_path)
  for ele in tfds:

    print("=====TFRecordDataset : {}".format(ele)) 
  


if __name__ == '__main__':
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from tensorflowonspark import TFParallel

  nxp_spark_conf = SparkConf()
  nxp_spark_conf.setAppName("NXP_inference")
  nxp_spark_conf.set('spark.executor.memory', '4g')
  nxp_spark_conf.set('spark.executor.cores', 4)
  nxp_spark_conf.set('spark.executor.instances',2)
  

  sc = SparkContext(conf=nxp_spark_conf.setAppName("mnist_inference"))
  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--cluster_size", help="number of nodes in the cluster (for S with labelspark Standalone)", type=int, default=num_executors)
  parser.add_argument('--images_labels', type=str, help='Directory for input images with labels', default="${TFoS_HOME}/data/mnist/tfr/test")
  parser.add_argument("--export_dir", help="HDFS path to export model", type=str, default=model_path)
  parser.add_argument("--output", help="HDFS path to save predictions", type=str, default="/usr/local/TensorFlowOnSpark/Zabbix_24hr_prediction/TFoS_24hr_Server_failure_prediction/predictions")
  args, _ = parser.parse_known_args()
  print("args: {}".format(args))

  # Running single-node TF instances on each executor
  TFParallel.run(sc, inference, args, args.cluster_size)
