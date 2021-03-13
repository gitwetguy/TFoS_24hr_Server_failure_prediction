from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import datetime

import argparse
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFParallel
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from tensorflowonspark import dfutil
from tensorflowonspark.pipeline import TFEstimator, TFModel

ATKH_mem_used_csv_path = "hdfs://master:9000/user/data/ATKH_Oplus_TWGKHHPSK1MSB04_memory_usage_2020_10.csv"
ATKH_mem_used_model_path = "/usr/local/TensorFlowOnSpark/Zabbix_24hr_prediction/TFoS_24hr_Server_failure_prediction/Zabbix"
#ATKH_mem_used_model_path = "hdfs://master:9000/user/data/Zabbix"
ATKH_mem_used_output_path = "/usr/local/TensorFlowOnSpark/Zabbix_24hr_prediction/TFoS_24hr_Server_failure_prediction/predictions"

def inference(args, ctx):
    
  print("inference")
    

    

    


if __name__ == '__main__':
  import argparse
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from tensorflowonspark import TFParallel
  from pyspark.sql import SparkSession
  from pyspark.sql.functions import udf
  from pyspark.sql.types import IntegerType
  from tensorflowonspark import dfutil
  from tensorflowonspark.pipeline import TFEstimator, TFModel
  
  nxp_spark_conf = SparkConf()
  nxp_spark_conf.setAppName("NXP_inference")
  nxp_spark_conf.set('spark.executor.memory', '4g')
  nxp_spark_conf.set('spark.executor.cores', 4)
  nxp_spark_conf.set('spark.executor.instances',2)
  sc = SparkContext(conf=nxp_spark_conf)
  spark = SparkSession(sc)
  executors = sc._conf.get("spark.executor.instances")
  print("=====executors = {}".format(executors))
  num_executors = int(executors) if executors is not None else 1
  print("=====num_executors = {}".format(num_executors))
  """ parser = argparse.ArgumentParser()
  parser.add_argument("--cluster_size", help="number of nodes in the cluster (for S with labelspark Standalone)", type=int, default=num_executors)
  #parser.add_argument('--images_labels', type=str, help='Directory for input images with labels')
  #parser.add_argument("--export_dir", help="HDFS path to export model", type=str, default="mnist_export")
  parser.add_argument("--output", help="HDFS path to save predictions", type=str, default="/usr/local/TensorFlowOnSpark/prediction")
  args, _ = parser.parse_known_args()
  print("args: {}".format(args)) """

  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=60)
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
  parser.add_argument("--format", help="example format: (csv|tfr)", choices=["csv", "tfr"], default="csv")
  parser.add_argument("--images_labels", help="path to MNIST images and labels in parallelized format")
  parser.add_argument("--mode", help="train|inference", choices=["train", "inference"], default="inference")
  parser.add_argument("--model_dir", help="path to save checkpoint", default=ATKH_mem_used_model_path)
  parser.add_argument("--export_dir", help="path to export saved_model", default=ATKH_mem_used_model_path)
  parser.add_argument("--output", help="HDFS path to save predictions", type=str, default=ATKH_mem_used_output_path)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true",default=True)
  args = parser.parse_args()
  print("args:", args)

  mem_data = spark.read.csv(ATKH_mem_used_csv_path,inferSchema=True, header=True)
  mem_data = mem_data.head(480)
  mem_data = spark.createDataFrame(mem_data)
  mem_data.show(60)

  model = TFModel(args) \
        .setInputMapping({'%used': 'gru_16_input'}) \
        .setOutputMapping({'dense_6': 'prediction'}) \
        .setSignatureDefKey('serving_default') \
        .setExportDir(args.export_dir) \
        .setBatchSize(args.batch_size)

  preds = model.transform(mem_data)
  preds.show()
  
  
  
  output_dir = datetime.datetime.now().strftime('%Y%m%d%H%M%S-prediction')
  tf.io.gfile.makedirs(ATKH_mem_used_output_path+"/"+output_dir)
  print("=======output_path : "+ATKH_mem_used_output_path+"/"+output_dir)
  preds.write.json(ATKH_mem_used_output_path+"/"+output_dir)
  result_json = spark.read.json(ATKH_mem_used_output_path+"/"+output_dir+"/*.json")
  result_json.show()
  #preds.write.json("predictions_res")
  #preds.write.json()


  #preds.write.format('json').save(ATKH_mem_used_output_path+"/"+output_dir)

 

  # Running single-node TF instances on each executor
  TFParallel.run(sc, inference, args, args.cluster_size)
