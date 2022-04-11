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
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # find key_values
    if isinstance(data, dict):
        key_values = find_all_keys_values(data,"")
    elif isinstance(data, list):
        key_values = find_all_keys_values({"data":data},"")
    else:
        raise ValueError('Your input JsonData is not a dictionary or list')

    key_values = {k:v for k,v in key_values.items() if len(v)>1}

    df = pd.DataFrame({k:pd.Series(v) for k,v in key_values.items()})
    # save to csv
    save_pth = re.sub(r'\.jsonl?','.csv',file_path)
    df.to_csv(save_pth, index=False, encoding='utf-8')