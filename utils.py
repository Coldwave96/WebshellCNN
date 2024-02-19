import os
import re
import math
import nltk

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
