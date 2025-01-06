"""
In this code we will be processing the raw dataset and converting it into a format that can be used for training the model.
no change has been made to the query.
"""

import json
import os
import pandas as pd


dataset_path = "data_processing/spider"
train_file = os.path.join(dataset_path, "train_spider.json")
dev_file = os.path.join(dataset_path, "dev.json")


train_df = pd.read_json(train_file)
dev_df = pd.read_json(dev_file)

del train_df['sql']
del dev_df['sql']
del train_df['query_toks_no_value']
del dev_df['query_toks_no_value']

train_df.to_json("data_processing/data/raw_train_data.json")
dev_df.to_json("data_processing/data/raw_dev_data.json")







