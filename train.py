import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import utils

parser = argparse.ArgumentParser(description="Train WebshellCNN model")

# Version args
default_version = "v1"
parser.add_argument("--config.version", type=str, default=default_version, help="Version")

# Data args
default_webshell_folder = "Dataset/webshell"
parser.add_argument("--data.webshell_folder", type=str, default=default_webshell_folder, help="Path to the folder which contains webshells")

default_normal_folder = "Dataset/normal"
parser.add_argument("--data.normal_folder", type=str, default=default_normal_folder, help="Path to the folder which containd normal files")

default_file_extensions = ['.php', '.asp', '.aspx', '.jsp', '.java']
parser.add_argument("--data.file_extensions", type=list, default=default_file_extensions, help=f"File extension list for training (default: {default_file_extensions})")

FLAGS = vars(parser.parse_args())

for key, value in FLAGS.items():
    print("{}={}".format(key, value))

processed_data_path = f'Processed_data/precessed_data_{FLAGS["config.version"]}.csv'
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

scaler_path = f'Output/Scaler/scaler_{FLAGS["config.version"]}.joblib'
if not os.path.exists(scaler_path):
    scaler = StandardScaler()
    data_df[['file_size', 'entropy']] = scaler.fit_transform(data_df[['file_size', 'entropy']])
    dump(scaler, scaler_path)
else:
    scaler = load(scaler_path)
    data_df[['file_size', 'entropy']] = scaler.transform(data_df[['file_size', 'entropy']])

max_features = 50000
sequence_length = 1024
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
text_vectorizer.adapt(data_df['word_list'].values)

text_vectorizer_config = text_vectorizer.get_config()
vectorizer_config_path = f'Output/TextVectorizer/text_vectorizer_config_{FLAGS["config.version"]}.json'
with open(vectorizer_config_path, 'w') as f:
    json.dump(text_vectorizer_config, f)

text_vectorizer_weights = text_vectorizer.get_weights()
vectorizer_weights_path = f'Output/TextVectorizer/text_vectorizer_weights_{FLAGS["config.version"]}.npy'
np.save(vectorizer_weights_path, text_vectorizer_weights)

text_vectorized = text_vectorizer(data_df['word_list'].values).numpy()

X_textcnn_train, X_textcnn_test, X_classification_train, X_classification_test, y_train, y_test = train_test_split(
    text_vectorized, 
    data_df[['file_size', 'entropy']].values, 
    data_df['label'].values, 
    test_size = 0.2,
    random_state = 23
)

model = utils.build_model(sequence_length, 2, max_features, 300)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit([X_textcnn_train, X_classification_train], y_train, epochs=5, batch_size=32, validation_split=0.2)

y_pred = model.predict([X_textcnn_test, X_classification_test])
y_pred_binary = np.round(y_pred)
report = classification_report(y_test, y_pred_binary)
print(report)

model_weights_path = f'Output/Model/combined_model_weights_{FLAGS["config.version"]}.h5'
model.save_weights(model_weights_path)
