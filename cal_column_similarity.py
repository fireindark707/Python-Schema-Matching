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
parser.add_argument("-t", "--threshold", help="threshold for inference")
parser.add_argument("-s", "--strategy", help="one-to-one or one-to-many", default="one-to-many")
args = parser.parse_args()

def create_similarity_matrix(pth,preds,pred_labels_list,strategy="one-to-many"):
    """
    Create a similarity matrix from the prediction
    """
    preds = np.array(preds)
    preds = np.mean(preds,axis=0)
    pred_labels_list = np.array(pred_labels_list)
    pred_labels = np.mean(pred_labels_list,axis=0)
    pred_labels = np.where(pred_labels>0.5,1,0)
    # read column names
    df1 = pd.read_csv(pth+"/Table1.csv")
    df2 = pd.read_csv(pth+"/Table2.csv")
    df1_cols = df1.columns
    df2_cols = df2.columns
    # create similarity matrix for pred values 
    preds_matrix = np.array(preds).reshape(len(df1_cols),len(df2_cols))
    if strategy == "one-to-many":
        pred_labels_matrix = np.array(pred_labels).reshape(len(df1_cols),len(df2_cols))
    elif strategy == "one-to-one":
        pred_labels_matrix = np.zeros((len(df1_cols),len(df2_cols)))
        for i in range(len(df1_cols)):
            for j in range(len(df2_cols)):
                if pred_labels[i*len(df2_cols)+j] == 1:
                    max_row = max(preds_matrix[i,:])
                    max_col = max(preds_matrix[:,j])
                    if preds_matrix[i,j] == max_row and preds_matrix[i,j] == max_col:
                        pred_labels_matrix[i,j] = 1
    df_pred = pd.DataFrame(preds_matrix,columns=df2_cols,index=df1_cols)
    df_pred_labels = pd.DataFrame(pred_labels_matrix,columns=df2_cols,index=df1_cols)
    return df_pred,df_pred_labels

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
        if args.threshold is not None:
            best_threshold = float(args.threshold)
        else:
            with open(model_pth+"/"+str(i)+".threshold",'r') as f:
                best_threshold = float(f.read())
        pred,pred_labels = test(bst,best_threshold,features,test_labels=np.ones(len(features)),type="inference")
        preds.append(pred)
        pred_labels_list.append(pred_labels)
        del bst

    df_pred,df_pred_labels = create_similarity_matrix(pth,preds,pred_labels_list,strategy=args.strategy)
    df_pred.to_csv(pth+"/similarity_matrix_value.csv",index=True)
    df_pred_labels.to_csv(pth+"/similarity_matrix_label.csv",index=True)