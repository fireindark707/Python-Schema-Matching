from . import init
from .self_features import make_self_features_from
from .utils import table_column_filter
import pandas as pd
import numpy as np
from numpy.linalg import norm
import random
import os
import subprocess
from strsimpy.metric_lcs import MetricLCS
from strsimpy.damerau import Damerau
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from sentence_transformers import util
import re

model = init.model

smoothie = SmoothingFunction().method4
metriclcs = MetricLCS()
damerau = Damerau()
seed = 200
random.seed(seed)

def preprocess_text(text):
    text = text.lower()
    text = re.split(r'[\s\_\.]', text)
    text = " ".join(text).strip()
    return text

def transformer_similarity(text1, text2):
    """
    Use sentence transformer to calculate similarity between two sentences.
    """
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    cosine_similarity = util.cos_sim(embeddings1, embeddings2)
    return cosine_similarity

def read_mapping(mapping_file):
    """
    Read mapping file and return a set.
    """
    if mapping_file is None or not os.path.exists(mapping_file):
        return set()
    with open(mapping_file, 'r') as f:
        readed = f.readlines()
    readed = [x.strip() for x in readed]
    mapping = set()
    for map in readed:
        map = map.split(",")
        map = [m.strip("< >") for m in map]
        mapping.add(tuple(map))
    return mapping

def make_combinations_labels(columns1, columns2, mapping ,type="train"):
    """
    Make combinations from columns1 list and columns2 list. Label them using mapping.
    """
    labels = {}
    for i,c1 in enumerate(columns1):
        for j,c2 in enumerate(columns2):
            if (c1, c2) in mapping or (c2, c1) in mapping:
                labels[(i, j)] = 1
            else:
                labels[(i, j)] = 0
    # sample negative labels
    if type == "train":
        combinations_count = len(labels)
        for i in range(combinations_count*2):
            if sum(labels.values()) >= 0.1 * len(labels):
                break
            c1 = random.choice(range(len(columns1)))
            c2 = random.choice(range(len(columns2)))
            if (c1, c2) in labels and labels[c1, c2] == 0:
                del labels[(c1, c2)]
    return labels

def get_colnames_features(text1,text2,column_name_embeddings):
    """
    Use BLEU, edit distance and word2vec to calculate features.
    """
    bleu_score = bleu([text1], text2, smoothing_function=smoothie)
    edit_distance = damerau.distance(text1, text2)
    lcs = metriclcs.distance(text1, text2)
    transformer_score = util.cos_sim(column_name_embeddings[text1], column_name_embeddings[text2])
    one_in_one = text1 in text2 or text2 in text1
    colnames_features = np.array([bleu_score, edit_distance, lcs,transformer_score, one_in_one])
    return colnames_features

def get_instance_similarity(embeddings1, embeddings2):
    """
    Use cosine similarity between two sentences.
    """
    cosine_similarity = np.inner(embeddings1, embeddings2) / (norm(embeddings1) * norm(embeddings2))
    return np.array([cosine_similarity])
    
def make_data_from(table1_df, table2_df,mapping_file=None,type="train"):
    """
    Read data from 2 table dataframe, mapping file path and make relational features and labels as a matrix.
    """
    mapping = read_mapping(mapping_file)
    columns1 = list(table1_df.columns)
    columns2 = list(table2_df.columns)

    combinations_labels = make_combinations_labels(columns1, columns2, mapping,type)
    table1_features = make_self_features_from(table1_df)
    table2_features = make_self_features_from(table2_df)

    column_name_embeddings = {preprocess_text(k):model.encode(preprocess_text(k)) for k in columns1+columns2}

    additional_feature_num = 6
    output_feature_table = np.zeros((len(combinations_labels), table1_features.shape[1] - 768+ additional_feature_num), dtype=np.float32)
    output_labels = np.zeros(len(combinations_labels), dtype=np.int32)
    for i, (combination,label) in enumerate(combinations_labels.items()):
        c1,c2 = combination
        c1_name = columns1[c1]
        c2_name = columns2[c2]
        difference_features_percent = np.abs(table1_features[c1] - table2_features[c2]) / (table1_features[c1] + table2_features[c2] + 1e-8)
        c1_name = preprocess_text(c1_name)
        c2_name = preprocess_text(c2_name)
        colnames_features = get_colnames_features(c1_name, c2_name,column_name_embeddings)
        instance_similarity = get_instance_similarity(table1_features[c1][-768:], table2_features[c2][-768:])
        output_feature_table[i,:] = np.concatenate((difference_features_percent[:-768], colnames_features, instance_similarity))
        output_labels[i] = label
        # add column names mask for training data
        if type == "train" and i % 5 == 0:
            colnames_features = np.array([0,12,0,0.2,0])
            added_features = np.concatenate((difference_features_percent[:-768], colnames_features, instance_similarity))
            added_features = added_features.reshape((1, added_features.shape[0]))
            output_feature_table = np.concatenate((output_feature_table, added_features), axis=0)
            output_labels = np.concatenate((output_labels, np.array([label])))
    return output_feature_table, output_labels

if __name__ == '__main__':
    if os.path.exists("Input"):
        #remove existing Input folder
        subprocess.call(["rm", "-r", "Input"])
    # make folders
    os.mkdir("Input")

    folder_list = os.listdir("Training Data")

    train_features = {}
    train_labels = {}
    test_features = {}
    test_labels = {}
    for folder in folder_list:
        print("start extracting data from " + folder)
        data_folder = "Training Data/" + folder
        table1_df = pd.read_csv(data_folder + "/Table1.csv")
        table2_df = pd.read_csv(data_folder + "/Table2.csv")
        table1_df = table_column_filter(table1_df)
        table2_df = table_column_filter(table2_df)
        mapping_file = data_folder + "/mapping.txt"
        features,labels = make_data_from(table1_df, table2_df, mapping_file,type="train")
        train_features[folder] = features
        train_labels[folder] = labels
        features,labels = make_data_from(table1_df, table2_df, mapping_file,type="test")
        test_features[folder] = features
        test_labels[folder] = labels

    # save data using cross validation
    for i in range(len(folder_list)):
        os.mkdir("Input/" + str(i))
        os.mkdir("Input/" + str(i) + "/train")
        os.mkdir("Input/" + str(i) + "/test")
        test_folder = folder_list[i]
        train_folders = folder_list[:i] + folder_list[i+1:]
        for folder in train_folders:
            np.save("Input/"+ str(i) +"/train/" +folder.split('/')[-1]+ "_features.npy", train_features[folder])
            np.save("Input/"+ str(i) +"/train/" +folder.split('/')[-1]+ "_labels.npy", train_labels[folder])
        np.save("Input/"+ str(i) +"/test/" +test_folder.split('/')[-1]+ "_features.npy", test_features[test_folder])
        np.save("Input/"+ str(i) +"/test/" +test_folder.split('/')[-1]+ "_labels.npy", test_labels[test_folder])
        
