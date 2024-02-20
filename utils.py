import os
import re
import math
import nltk
import tensorflow as tf

def list_files(folder_path, file_extensions):
    file_path_list = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                file_path_list.append(file_path)
    
    return file_path_list

def remove_comments(file_path):
    single_line_pattern1 = re.compile(r'//.*')
    single_line_pattern2 = re.compile(r'#.*')
    multi_line_pattern = re.compile(r'/\*[\s\S]*?\*/')

    with open(file_path, 'r', encoding='gb18030', errors='ignore') as file:
        content = file.read()
        
        content = re.sub(single_line_pattern1, '', content)
        content = re.sub(single_line_pattern2, '', content)
        content = re.sub(multi_line_pattern, '', content)

    return content

def calculate_entropy(file_path):
    content = remove_comments(file_path)
    word_list = {}
    sum = 0
    entropy = 0
    for word in content:
        if word != '\n' and word != ' ':
            if word not in word_list.keys():
                word_list[word] = 1
            else:
                word_list[word] += 1
    
    for index in word_list.keys():
        sum += word_list[index]
    
    for index in word_list.keys():
        entropy -= float(word_list[index])/sum * math.log(float(word_list[index])/sum, 2)
    
    return entropy

def calculate_size(file_path):
    file_size = os.path.getsize(file_path)
    return file_size

def preprocess_file(file_path):
    content = remove_comments(file_path).lower()
    try:
        word_list = nltk.tokenize.word_tokenize(content)
    except LookupError:
        nltk.download('punkt', download_dir='./')

    punctuation_list = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', 'php', '<', '>', '\'', 'java', '{', '}']
    word_list = [word for word in word_list if word not in punctuation_list]

    return word_list

def create_textcnn_model(sequence_length, vocab_size, embedding_dim):
    word_list_input = tf.keras.layers.Input(shape=(sequence_length,), name='word_list_input')
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length)(word_list_input)

    conv1 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu')(embedding_layer)
    maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=int(conv1.shape[1]))(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=4, activation='relu')(embedding_layer)
    maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=int(conv2.shape[1]))(conv2)

    conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu')(embedding_layer)
    maxpool3 = tf.keras.layers.MaxPooling1D(pool_size=int(conv3.shape[1]))(conv3)

    concatenated = tf.keras.layers.Concatenate(axis=-1)([maxpool1, maxpool2, maxpool3])
    flattened = tf.keras.layers.Flatten()(concatenated)

    dense1 = tf.keras.layers.Dense(128)(flattened)
    dropout = tf.keras.layers.Dropout(0.25)(dense1)
    dense1_relu = tf.keras.layers.ReLU()(dropout)
    output_textcnn = tf.keras.layers.Dense(1, activation='sigmoid')(dense1_relu)

    textcnn_model = tf.keras.Model(inputs=word_list_input, outputs=output_textcnn, name='textcnn_model')
    return textcnn_model

def create_classification_model(input_dim):
    file_info_input = tf.keras.layers.Input(shape=(input_dim,), name='file_info_input')
    dense1 = tf.keras.layers.Dense(64, activation='relu')(file_info_input)
    dropout = tf.keras.layers.Dropout(0.25)(dense1)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout)
    output_classifier = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

    classification_model = tf.keras.Model(inputs=file_info_input, outputs=output_classifier, name='calssification_model')
    return classification_model

def build_model(sequence_length, classidication_input_dim, vocab_size, embedding_dim):
    textcnn_model = create_textcnn_model(sequence_length, vocab_size, embedding_dim)
    classification_model = create_classification_model(classidication_input_dim)

    merged_model = tf.keras.layers.Concatenate()([textcnn_model.output, classification_model.output])
    dense = tf.keras.layers.Dense(32, activation='relu')(merged_model)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='final_output')(dense)

    model = tf.keras.Model(inputs=[textcnn_model.input, classification_model.input], outputs=output_layer, name='combined_model')
    return model
