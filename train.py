import os
import argparse
import pandas as pd
import tensorflow as tf

import utils

parser = argparse.ArgumentParser(description="Train WebshellCNN model")

# Dara args
default_webshell_folder = "Dataset/webshell"
parser.add_argument("--data.webshell_folder", type=str, default=default_webshell_folder, help="Path to the folder which contains webshells")

default_normal_folder = "Dataset/normal"
parser.add_argument("--data.normal_folder", type=str, default=default_normal_folder, help="Path to the folder which containd normal files")

default_file_extensions = ['.php', '.asp', '.aspx', '.jsp', '.java']
parser.add_argument("--data.file_extensions", type=list, default=default_file_extensions, help=f"File extension list for training (default: {default_file_extensions})")

FLAGS = vars(parser.parse_args())

for key, value in FLAGS.items():
    print("{}={}".format(key, value))

processed_data_path = f'Processed_data/{FLAGS["data.webshell_folder"].replace("/", "-")}_{FLAGS["data.normal_folder"].replace("/", "-")}.csv'
if not os.path.exists(processed_data_path):
    webshell_path_list = utils.list_files(FLAGS["data.webshell_folder"], FLAGS["data.file_extensions"])
    normal_path_list = utils.list_files(FLAGS["data.normal_folder"], FLAGS["data.file_extensions"])

    webshell_df = pd.DataFrame(webshell_path_list, columns=['file'])
    normal_df = pd.DataFrame(normal_path_list, columns=['file'])
    webshell_df['label'] = 1
    normal_df['label'] = 0
    data_df = pd.concat([webshell_df, normal_df], ignore_index=True)

    data_df['file_size'] = data_df['file'].apply(utils.calculate_size)
    data_df['entropy'] = data_df['file'].apply(utils.calculate_entropy)
    data_df['word_list'] = data_df['file'].apply(utils.preprocess_file)

    data_df.to_csv(processed_data_path)
else:
    data_df = pd.read_csv(processed_data_path)

dataset = tf.data.Dataset.from_tensor_slices((data_df['word_list'].values, data_df['label'].values))

max_features = 50000
sequence_length = 1024
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length,
)

vectorize_layer.adapt(data_df['word_list'].values)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

vectorized_data = dataset.map(vectorize_text)

for example in vectorized_data.take(1):
    print("Vectorized Word List:", example[0].numpy())
    print("Label:", example[1].numpy())
