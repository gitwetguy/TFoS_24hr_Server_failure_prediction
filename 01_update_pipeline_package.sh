#!/bin/bash

#update pipeline.py on slave node (worker node)

scp /home/master/anaconda3/lib/python3.7/site-packages/tensorflowonspark/pipeline.py  slave1:/home/master/anaconda3/lib/python3.7/site-packages/tensorflowonspark

cp /home/master/anaconda3/lib/python3.7/site-packages/tensorflowonspark/pipeline.py /usr/local/TensorFlowOnSpark/Zabbix_24hr_prediction/TFoS_24hr_Server_failure_prediction/Zabbix_specify_pipeline.py

