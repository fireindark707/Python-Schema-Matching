from relation_features import make_data_from
from train import test
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path", help="path to the folder containing the test data")
parser.add_argument("-m", "--model", help="path to the model")
args = parser.parse_args()

def create_similarity_matrix(pth,pred):
    """
    Create a similarity matrix from the prediction
    """
    df1 = pd.read_csv(pth+"/Table1.csv")
    df2 = pd.read_csv(pth+"/Table2.csv")
    df1_cols = df1.columns
    df2_cols = df2.columns
    # create similarity matrix for pred values
    sim_matrix = np.zeros((len(df1_cols),len(df2_cols)))
    for i in range(len(df1_cols)):
        for j in range(len(df2_cols)):
            sim_matrix[i,j] = pred[i*len(df2_cols)+j]
    # create dataframe
    df = pd.DataFrame(sim_matrix,index=df1_cols,columns=df2_cols)
    return df

if __name__ == '__main__':
    pth = args.path
    model_pth = args.model

    features,_ = make_data_from(pth,"test")
    preds = []
    pred_labels_list = []
    for i in range(len(os.listdir(model_pth))//2):
        print("start using model " + str(i))
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(model_pth+"/"+str(i)+".model")
        with open(model_pth+"/"+str(i)+".threshold",'r') as f:
            best_threshold = float(f.read())
        pred,pred_labels = test(bst,best_threshold,features,test_labels=np.ones(len(features)),type="inference")
        preds.append(pred)
        pred_labels_list.append(pred_labels)
        del bst
    preds = np.array(preds)
    preds = np.mean(preds,axis=0)
    pred_labels_list = np.array(pred_labels_list)
    pred_labels = np.mean(pred_labels_list,axis=0)
    pred_labels = np.where(pred_labels>0.5,1,0)

    df_pred = create_similarity_matrix(pth,pred)
    df_pred_labels = create_similarity_matrix(pth,pred_labels)
    df_pred.to_csv(pth+"/similarity_matrix_value.csv")
    df_pred_labels.to_csv(pth+"/similarity_matrix_label.csv")