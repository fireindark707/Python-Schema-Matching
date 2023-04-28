import init
from relation_features import make_data_from
from utils import make_csv_from_json,table_column_filter
from train import test
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import argparse
import time
from pathlib import Path

this_directory = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path", help="path to the folder containing the test data")
parser.add_argument("-m", "--model", help="path to the model")
parser.add_argument("-t", "--threshold", help="threshold for inference")
parser.add_argument("-s", "--strategy", help="one-to-one or many-to-many or one-to-many", default="many-to-many")
args = parser.parse_args()

def create_similarity_matrix(table1_df,table2_df,preds,pred_labels_list,strategy="many-to-many"):
    """
    Create a similarity matrix from the prediction
    """
    predicted_pairs = []
    preds = np.array(preds)
    preds = np.mean(preds,axis=0)
    pred_labels_list = np.array(pred_labels_list)
    pred_labels = np.mean(pred_labels_list,axis=0)
    pred_labels = np.where(pred_labels>0.5,1,0)
    # read column names
    df1_cols = table1_df.columns
    df2_cols = table2_df.columns
    # create similarity matrix for pred values 
    preds_matrix = np.array(preds).reshape(len(df1_cols),len(df2_cols))
    # create similarity matrix for pred labels
    if strategy == "many-to-many":
        pred_labels_matrix = np.array(pred_labels).reshape(len(df1_cols),len(df2_cols))
    else:
        pred_labels_matrix = np.zeros((len(df1_cols),len(df2_cols)))
        for i in range(len(df1_cols)):
            for j in range(len(df2_cols)):
                if pred_labels[i*len(df2_cols)+j] == 1:
                    if strategy == "one-to-one":
                        max_row = max(preds_matrix[i,:])
                        max_col = max(preds_matrix[:,j])
                        if preds_matrix[i,j] == max_row and preds_matrix[i,j] == max_col:
                            pred_labels_matrix[i,j] = 1
                    elif strategy == "one-to-many":
                        max_row = max(preds_matrix[i,:])
                        if preds_matrix[i,j] == max_row:
                            pred_labels_matrix[i,j] = 1
    df_pred = pd.DataFrame(preds_matrix,columns=df2_cols,index=df1_cols)
    df_pred_labels = pd.DataFrame(pred_labels_matrix,columns=df2_cols,index=df1_cols)
    for i in range(len(df_pred_labels)):
        for j in range(len(df_pred_labels.iloc[i])):
            if df_pred_labels.iloc[i,j] == 1:
                predicted_pairs.append((df_pred.index[i],df_pred.columns[j],df_pred.iloc[i,j]))
    return df_pred,df_pred_labels,predicted_pairs

def schema_matching(table1_pth,table2_pth,threshold=None,strategy="many-to-many",model_pth=None):
    """
    Do schema matching!
    """
    if model_pth is None:
        model_pth = str(this_directory / "model" / "2022-04-12-12-06-32")
    # transform jsonl or json file to csv
    if table1_pth.endswith('.json') or table1_pth.endswith('.jsonl'):
        table1_df = make_csv_from_json(table1_pth)
    else:
        table1_df = pd.read_csv(table1_pth)
    if table2_pth.endswith('.json') or table2_pth.endswith('.jsonl'):
        table2_df = make_csv_from_json(table2_pth)
    else:
        table2_df = pd.read_csv(table2_pth)

    # filter columns
    table1_df = table_column_filter(table1_df)
    table2_df = table_column_filter(table2_df)

    # extract features
    features,_ = make_data_from(table1_df, table2_df, type="test")

    # load model and predict on features
    preds = []
    pred_labels_list = []
    for i in range(len(os.listdir(model_pth))//2):
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(model_pth+"/"+str(i)+".model")
        if threshold is not None:
            best_threshold = float(threshold)
        else:
            with open(model_pth+"/"+str(i)+".threshold",'r') as f:
                best_threshold = float(f.read())
        pred, pred_labels = test(bst, best_threshold, features, test_labels=np.ones(len(features)), type="inference")
        preds.append(pred)
        pred_labels_list.append(pred_labels)
        del bst

    df_pred,df_pred_labels,predicted_pairs = create_similarity_matrix(table1_df, table2_df, preds, pred_labels_list, strategy=strategy)
    return df_pred,df_pred_labels,predicted_pairs

if __name__ == '__main__':
    start = time.time()
    args.path = args.path.rstrip("/")
    df_pred,df_pred_labels,predicted_pairs = schema_matching(args.path+"/Table1.csv",args.path+"/Table2.csv",threshold=args.threshold,strategy=args.strategy,model_pth=args.model)
    df_pred.to_csv(args.path+"/similarity_matrix_value.csv",index=True)
    df_pred_labels.to_csv(args.path+"/similarity_matrix_label.csv",index=True)

    for pair_tuple in predicted_pairs:
        print(pair_tuple)
    print("schema_matching|Time taken:",time.time()-start)