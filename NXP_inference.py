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
    

def resample(column, agg_interval=900, time_format='yyyy-MM-dd HH:mm:ss'):
    if type(column)==str:
        column = F.col(column)

    # Convert the timestamp to unix timestamp format.
    # Unix timestamp = number of seconds since 00:00:00 UTC, 1 January 1970.
    col_ut =  F.unix_timestamp(column, format=time_format)

    # Divide the time into dicrete intervals, by rounding. 
    col_ut_agg =  F.floor(col_ut / agg_interval) * agg_interval  

    # Convert to and return a human readable timestamp
    return F.from_unixtime(col_ut_agg)    

    


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
  import time
  from pyspark.sql.functions import udf
  import matplotlib.pyplot as plt
  

  executor_core = 4
  executor_instances = 2

  nxp_spark_conf = SparkConf()
  nxp_spark_conf.setAppName("NXP_inference")
  nxp_spark_conf.set('spark.executor.memory', '8g')
  nxp_spark_conf.set('spark.executor.cores', executor_core)
  nxp_spark_conf.set('spark.executor.instances',executor_instances)
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

  proccess_time_list = []
  for i in [480*30]:
    
    start = time.process_time()
    mem_data = spark.read.csv(ATKH_mem_used_csv_path,inferSchema=True, header=True)
    
    #mergeCols = udf(lambda Date, time: Date + time)
    #mem_data.withColumn("dt", mergeCols(col("Date"), col("time"))).show(1,False)


    
    

    mem_data = mem_data.head(i)
    mem_data = spark.createDataFrame(mem_data)
    #mem_data.resample('H')

    #mem_data.show(60)

    model = TFModel(args) \
          .setInputMapping({'%used': 'gru_16_input'}) \
          .setOutputMapping({'dense_6': 'prediction'}) \
          .setSignatureDefKey('serving_default') \
          .setExportDir(args.export_dir) \
          .setBatchSize(args.batch_size)

    preds = model.transform(mem_data)
    #preds.show()

    end = time.process_time()  
    proccess_time_list.append(end-start)  
      
    output_dir = datetime.datetime.now().strftime('%Y%m%d%H%M%S-prediction')
    tf.io.gfile.makedirs(ATKH_mem_used_output_path+"/"+output_dir)
    #print("=======output_path : "+ATKH_mem_used_output_path+"/"+output_dir)
    preds.write.json(ATKH_mem_used_output_path+"/"+output_dir,mode='append')
    result_json = spark.read.json(ATKH_mem_used_output_path+"/"+output_dir+"/*.json")
      
    with open(ATKH_mem_used_output_path+"/"+output_dir+'/'+output_dir+'.txt', 'w') as f:
      for item in result_json.collect():
          if item[0].__getitem__(0) >= 50:
            print("\nWarning! Anomaly memory usage rate is predicted : 2020/10/26-21:42:45 : {}\n".format(item[0].__getitem__(0)))
          f.write("%s\n" % item)
  
  print("=====Proccess_time : {}".format(proccess_time_list))
  #plt.plot(proccess_time_list)
  #plt.show()
  
  #with open("/usr/local/TensorFlowOnSpark/Zabbix_24hr_prediction/TFoS_24hr_Server_failure_prediction/proccess_time.txt")
 

  # Running single-node TF instances on each executor
  TFParallel.run(sc, inference, args, args.cluster_size)
