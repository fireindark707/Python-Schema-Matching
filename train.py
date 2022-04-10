import pandas as pd
import numpy as np
import os
import xgboost as xgb
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
import time
import warnings
warnings.filterwarnings("ignore")

feature_names = ["is_url","is_numeric","is_date","is_string","numeric:mean", "numeric:min", "numeric:max", "numeric:variance","numeric:cv", "numeric:unique/len(data_list)",
                "length:mean", "length:min", "length:max", "length:variance","length:cv", "length:unique/len(data_list)",
                "whitespace_ratios:mean","punctuation_ratios:mean","special_character_ratios:mean","numeric_ratios:mean",
                "whitespace_ratios:cv","punctuation_ratios:cv","special_character_ratios:cv","numeric_ratios:cv",
                "colname:bleu_score", "colname:edit_distance","colname:lcs","colname:tsm_cosine","colname:tsm_dot", "colname:one_in_one", "colname:all_same"
                ]

params = {
        'max_depth': 4,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
    }

def train(train_features,train_labels,num_round=400):
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    bst = xgb.train(params, dtrain, num_round)
    # get best_threshold
    best_f1 = 0
    best_threshold = 0
    for threshold in range(100):
        threshold = threshold / 100
        pred_labels = np.where(bst.predict(dtrain) > threshold, 1, 0)
        f1 = f1_score(train_labels, pred_labels,average="binary",pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return bst,best_threshold

def test(bst,best_threshold, test_features, test_labels, type="evaluation"):
    dtest = xgb.DMatrix(test_features, label=test_labels)
    pred = bst.predict(dtest)
    if type == "inference":
        pred_labels = np.where(pred > best_threshold, 1, 0)
        return pred,pred_labels
    # compute precision, recall, and F1 score
    pred_labels = np.where(pred > best_threshold, 1, 0)
    precision = precision_score(test_labels, pred_labels,average="binary",pos_label=1)
    recall = recall_score(test_labels, pred_labels,average="binary",pos_label=1)
    f1 = f1_score(test_labels, pred_labels,average="binary",pos_label=1)
    c_matrix = confusion_matrix(test_labels, pred_labels)
    return precision, recall, f1, c_matrix

def merge_features(path):
    files = os.listdir(path)
    files.sort()
    merged_features = []
    for file in files:
        if not "features" in file:
            continue
        features = np.load(path + file)
        merged_features.append(features)
    return np.concatenate(merged_features)

def get_labels(path):
    files = os.listdir(path)
    files.sort()
    labels = []
    for file in files:
        if not "labels" in file:
            continue
        labels.append(np.load(path + file))
    return np.concatenate(labels)

def preprocess(path):
    train_path = path + "/train/"
    test_path = path + "/test/"

    train_features = merge_features(train_path)
    train_labels = get_labels(train_path)
    test_features = merge_features(test_path)
    test_labels = get_labels(test_path)

    return train_features, train_labels, test_features, test_labels

def get_feature_importances(bst):
    importance = bst.get_fscore()
    importance = [(im,feature_names[int(im[0].replace("f",""))]) for im in importance.items()]
    importance = sorted(importance, key=lambda x: x[0][1], reverse=True)
    return importance

if __name__ == '__main__':
    model_save_pth = "model/"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not os.path.exists(model_save_pth):
        os.makedirs(model_save_pth)
    precision_list = []
    recall_list = []
    f1_list = []
    c_matrix_list = []
    feature_importance_list = []
    for i in range(len(os.listdir("Input"))):
        time.sleep(1)
        train_features, train_labels, test_features, test_labels = preprocess("Input/" + str(i))
        bst, best_threshold = train(train_features, train_labels)
        precision, recall, f1, c_matrix = test(bst,best_threshold, test_features, test_labels)
        feature_importance = get_feature_importances(bst)
        #print(f"Positive rate in Training: {sum(train_labels)/len(train_labels)*100:.2f}%")
        #print(f"Positive rate in Testing: {sum(test_labels)/len(test_labels)*100:.2f}%")
        c_matrix_norm = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        c_matrix_list.append(c_matrix_norm)
        feature_importance_list.append(feature_importance)
        bst.save_model(model_save_pth+f"/{i}.model")
        with open(model_save_pth+f"/{i}.threshold",'w') as f:
            f.write(str(best_threshold))
    # give evaluation results
    print("Average Precision: %.2f" % np.mean(precision_list))
    print("Average Recall: %.2f" % np.mean(recall_list))
    print("Average F1: %.2f" % np.mean(f1_list))
    print(f1_list)
    print(np.mean(c_matrix_list,axis=0))
    # evaluate feature importance
    feature_name_importance = {}
    for feature_importance in feature_importance_list:
        for (im,feature_name) in feature_importance:
            if feature_name in feature_name_importance:
                feature_name_importance[feature_name] += im[1]
            else:
                feature_name_importance[feature_name] = im[1]
    feature_name_importance = sorted(feature_name_importance.items(), key=lambda x: x[1], reverse=True)
    print('feature importance:')
    for item in feature_name_importance:
        print(item)