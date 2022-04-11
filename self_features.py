import pandas as pd
import numpy as np
import re
from dateutil.parser import parse as parse_date
import random

unit_dict = {"万": 10000, "亿": 100000000, "萬": 10000, "億": 100000000, "K+": 1000, "M+": 1000000, "B+": 1000000000}

def load_table(filepath):
    """
    Loads the data from the given filepath.
    """
    df = pd.read_csv(filepath)
    return df

def strict_numeric(data_list,verbose=False):
    """
    Checks if the given data is numeric.
    """
    cnt = 0
    for x in data_list:
        try:
            y = float(x)
            if verbose:
                print(x)
                print(y)
            cnt += 1
        except:
            continue
    if cnt >= 0.95*len(data_list):
        return True
    return False

def mainly_numeric(data_list):
    """
    Checks if the given data list is mostly numeric.
    """
    cnt = 0
    for data in data_list:
        data = data.replace(",", "")
        for unit in unit_dict.keys():
            data = data.replace(unit, "")
        numeric_part = re.findall(r'\d+', data)
        if len(numeric_part) > 0 and sum(len(x) for x in numeric_part) >= 0.5*len(data):
            cnt += 1 
    if cnt >= 0.9*len(data_list):
        return True
    return False

def extract_numeric(data_list):
    """
    Extracts numeric part(including float) from string list
    """
    try:
        data_list = [float(d) for d in data_list]
    except:
        pass
    numeric_part = []
    unit = []
    for data in data_list:
        data = str(data)
        data = data.replace(",", "")
        numeric_part.append(re.findall(r'([-]?([0-9]*[.])?[0-9]+)', data))
        this_unit = 1
        for unit_key in unit_dict.keys():
            if unit_key in data:
                this_unit = unit_dict[unit_key]
                break
        unit.append(this_unit)
    numeric_part = [x for x in numeric_part if len(x) > 0]
    if len(numeric_part) != len(data_list):
        print(f"Warning: extract_numeric() found different number of numeric part({len(numeric_part)}) and data list({len(data_list)})")
    numeric_part = [float(x[0][0])*unit[i] for i,x in enumerate(numeric_part)]
    return numeric_part

def numeric_features(data_list):
    """
    Extracts numeric features from the given data. Including Mean,Min, Max, Variance, Standard Deviation,
    and the number of unique values.
    """
    mean = np.mean(data_list)
    min = np.min(data_list)
    max = np.max(data_list)
    variance = np.var(data_list)
    cv = np.var(data_list)/mean
    unique = len(set(data_list))
    return np.array([mean, min, max, variance,cv, unique/len(data_list)])

def is_url(data_list):
    """
    Checks if the given data is in URL format.
    """
    cnt = 0
    for data in data_list:
        if type(data) != str:
            continue
        if re.search(r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', data):
            cnt += 1
    if cnt >= 0.9*len(data_list):
        return True
    return False

def is_date(data_list):
    """
    Checks if the given data is in Date format.
    """
    cnt = 0
    for data in data_list:
        if type(data) != str:
            continue
        if "月" in data or "日" in data or "年" in data:
            cnt += 1
        try:
            date = parse_date(data)
            # check if the date is near to today
            if date.year < 2000 or date.year > 2030:
                continue
            cnt += 1
        except:
            continue
    if cnt >= 0.9*len(data_list):
        return True
    return False

def character_features(data_list):
    """
    Extracts character features from the given data. 
    """
    # Ratio of whitespace to length
    # Ratio of punctuation to length
    # Ratio of special characters to length
    punctuations = [",",".",";","!","?","，","。","；","！","？"]
    special_characters = ["／","/","\\","-","_","+","=","*","&","^","%","$","#","@","~","`","(",")","[","]","{","}","<",">","|","'","\""]
    whitespace_ratios = []
    punctuation_ratios = []
    special_character_ratios = []
    numeric_ratios = []
    for data in data_list:
        whitespace_ratio = (data.count(" ") + data.count("\t") + data.count("\n"))/len(data)
        punctuation_ratio = sum(1 for x in data if x in punctuations)/len(data)
        special_character_ratio = sum(1 for x in data if x in special_characters)/len(data)
        numeric_ratio = sum(1 for x in data if x.isdigit())/len(data)
        whitespace_ratios.append(whitespace_ratio)
        punctuation_ratios.append(punctuation_ratio)
        special_character_ratios.append(special_character_ratio)
        numeric_ratios.append(numeric_ratio)
    epilson = np.array([1e-12]*len(data_list))
    whitespace_ratios = np.array(whitespace_ratios + epilson)
    punctuation_ratios = np.array(punctuation_ratios + epilson)
    special_character_ratios = np.array(special_character_ratios + epilson)
    numeric_ratios = np.array(numeric_ratios + epilson)
    return np.array([np.mean(whitespace_ratios), np.mean(punctuation_ratios), np.mean(special_character_ratios), np.mean(numeric_ratios),
                     np.var(whitespace_ratios)/np.mean(whitespace_ratios), np.var(punctuation_ratios)/np.mean(punctuation_ratios),
                        np.var(special_character_ratios)/np.mean(special_character_ratios), np.var(numeric_ratios)/np.mean(numeric_ratios)])

def extract_features(data_list):
    """
    Extract some features from the given data(column) or list
    """
    data_list = [d for d in data_list if d == d and d != "--"]
    if len(data_list) == 0:
        return 0
    data_types = ("url","numeric","date","string")
    # Classify the data's type, URL or Date or Numeric
    if is_url(data_list):
        data_type = "url"
    elif is_date(data_list):
        data_type = "date"
    elif strict_numeric(data_list) or mainly_numeric(data_list):
        data_type = "numeric"
    else:
        data_type = "string"
    # Make data type feature one hot encoding
    data_type_feature = np.zeros(len(data_types))
    data_type_feature[data_types.index(data_type)] = 1
    # Give numeric features if the data is mostly numeric
    if data_type == "numeric": 
        data_numeric = extract_numeric(data_list)
        num_fts = numeric_features(data_numeric)
    else:
        num_fts = np.array([-1]*6)
    # If data is not numeric, give length features
    length_fts = numeric_features([len(str(d)) for d in data_list])
    # Give character features if the data is string
    if data_type == "string" or (not strict_numeric(data_list) and  mainly_numeric(data_list)):
        char_fts = character_features(data_list)
    else:
        char_fts = np.array([-1]*8)
    output_features = np.concatenate((data_type_feature, num_fts, length_fts, char_fts))
    return output_features

def make_self_features_from(filepath):
    """
    Extracts features from the given table path and returns a feature table.
    """
    df = load_table(filepath)
    features = None
    for column in df.columns:
        if "Unnamed:" in column:
            continue
        fts = extract_features(df[column])
        if type(fts) == int:
            continue
        fts = fts.reshape(1, -1)
        if features is None:
            features = fts
        else:
            features = np.concatenate((features, fts), axis=0)
    return features

if __name__ == '__main__':
    features = make_self_features_from("Training Data/pair_7/Table1.csv")
    print(features)