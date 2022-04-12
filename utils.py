import pandas as pd
import json
from collections import defaultdict
import re

def find_all_keys_values(json_data,parent_key):
    """
    Find all keys that don't have list or dictionary values and their values. Key should be saved with its parent key like "parent-key.key".
    """
    key_values = defaultdict(list)
    for key, value in json_data.items():
        if isinstance(value, dict):
            child_key_values = find_all_keys_values(value,key)
            for child_key, child_value in child_key_values.items():
                key_values[child_key].extend(child_value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    child_key_values = find_all_keys_values(item,key)
                    for child_key, child_value in child_key_values.items():
                        key_values[child_key].extend(child_value)
                else:
                    key_values[parent_key+"."+key].append(item)
        else:
            key_values[parent_key+"."+key].append(value)
    return key_values

def make_csv_from_json(file_path):
    """
    Make csv file from json file.
    """
    if file_path.endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.endswith(".jsonl"):
        data = []
        with open(file_path, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            data.append(json.loads(json_str))

    # find key_values
    if isinstance(data, dict):
        key_values = find_all_keys_values(data,"")
    elif isinstance(data, list):
        key_values = find_all_keys_values({"TOPLEVEL":data},"TOPLEVEL")
    else:
        raise ValueError('Your input JsonData is not a dictionary or list')

    key_values = {k.replace("TOPLEVEL.",""):v for k,v in key_values.items() if len(v)>1}

    df = pd.DataFrame({k:pd.Series(v) for k,v in key_values.items()})
    # save to csv
    save_pth = re.sub(r'\.jsonl?','.csv',file_path)
    df.to_csv(save_pth, index=False, encoding='utf-8')
    return df

def table_column_filter(table_df):
    """
    Filter columns that have zero instances or all columns are "--"
    """
    original_columns = table_df.columns
    for column in table_df.columns:
        column_data = [d for d in list(table_df[column]) if d == d and d != "--"]
        if len(column_data) <= 1:
            table_df = table_df.drop(column, axis=1)
            continue
        if "Unnamed:" in column:
            table_df = table_df.drop(column, axis=1)
            continue
    remove_columns = list(set(original_columns) - set(table_df.columns))
    if len(remove_columns) > 0:
        print("Removed columns:", remove_columns)
    return table_df