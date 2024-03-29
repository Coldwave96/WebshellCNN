import pickle
import argparse
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf

import utils

parser = argparse.ArgumentParser(description="Train WebshellCNN model")

# Version args
default_version = "v1"
parser.add_argument("--config.version", type=str, default=default_version, help="Version")

default_config_folder = "Output/"
parser.add_argument("--config.folder", type=str, default=default_config_folder, help=f"Folder which contains all config and weight of model/scaler/TextVectorizer")

# Data args
default_unknown_folder = "Dataset/predict"
parser.add_argument("--data.unknown_folder", type=str, default=default_unknown_folder, help="Folder to be detected")

default_file_extensions = ['.php', '.asp', '.aspx', '.jsp', '.java']
parser.add_argument("--data.file_extensions", type=list, default=default_file_extensions, help=f"File extension list for training (default: {default_file_extensions})")

# Predict args
default_max_features = 5000
parser.add_argument("--predict.max_features", type=int, default=default_max_features, help=f"Max tokens for TextVectorizer (default: {default_max_features})")

default_sequence_length = 1024
parser.add_argument("--predict.sequence_length", type=int, default=default_sequence_length, help=f"Output sequence length for TextVectorizer (default: {default_sequence_length})")

default_embedding_dim = 300
parser.add_argument("--predict.embedding_dim", type=int, default=default_embedding_dim, help=f"Ouput dimensions of the embedding layer (default: {default_embedding_dim})")

FLAGS = vars(parser.parse_args())

for key, value in FLAGS.items():
    print("{}={}".format(key, value))

unknown_folder_list = utils.list_files(FLAGS["data.unknown_folder"], FLAGS["data.file_extensions"])
unknown_df = pd.DataFrame(unknown_folder_list, columns=['file'])
unknown_df['file_size'] = unknown_df['file'].apply(utils.calculate_size)
unknown_df['entropy'] = unknown_df['file'].apply(utils.calculate_entropy)
unknown_df['word_list'] = unknown_df['file'].apply(utils.preprocess_file)

scaler_path = f'{FLAGS["config.folder"]}Scaler/scaler_{FLAGS["config.version"]}.joblib'
scaler = load(scaler_path)
unknown_df[['file_size', 'entropy']] = scaler.transform(unknown_df[['file_size', 'entropy']])

vocabulary = []
vectorizer_vocab_path = f'{FLAGS["config.folder"]}TextVectorizer/text_vectorizer_vocab_{FLAGS["config.version"]}.pkl'
with open(vectorizer_vocab_path, 'rb') as f:
    vocabulary = pickle.load(f)

max_features = FLAGS["predict.max_features"]
sequence_length = FLAGS["predict.sequence_length"]

text_vectorizer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=sequence_length)
text_vectorizer.set_vocabulary(vocabulary)

x_textcnn_unknown = [' '.join(sublist) for sublist in unknown_df['word_list'].values]
x_textcnn_unknown = text_vectorizer(x_textcnn_unknown)

x_file_info_unknown = unknown_df[['file_size', 'entropy']].values

embedding_dim = FLAGS["predict.embedding_dim"]
model_weights_path = f'{FLAGS["config.folder"]}Model/combined_model_weights_{FLAGS["config.version"]}.h5'
model = utils.build_model(sequence_length, 2, max_features, embedding_dim)
model.load_weights(model_weights_path)

predictions = model.predict([x_textcnn_unknown, x_file_info_unknown])
unknown_df['predict'] = predictions
print(unknown_df)
