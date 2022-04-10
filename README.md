# Schema Matching by XGboost
Using XGboost to perform schema matching task on tables. Support multi-language column names matching and can be used without column names.

## Data

See Data format in Training Data and Test Data folders. You need to put mapping.txt, Table1.csv and Table2.csv in new folders under Training Data. For Test Data, mapping.txt is not needed.

## Output

- similarity_matrix_label.csv: Labels(0,1) for each column pairs.
- similarity_matrix_value.csv: Average of raw values computed by all the xgboost models.

## Usage

### 1.Construct features

python relation_features.py

### 2.Train xgboost models

python train.py

### 3.Calculate similarity matrix (inference)

Example: python cal_column_similarity.py -p Test\ Data/self -m model/2022-04-10-20-48-57

## Mechanism

Features: "is_url","is_numeric","is_date","is_string","numeric:mean", "numeric:min", "numeric:max", "numeric:variance","numeric:cv", "numeric:unique/len(data_list)", "length:mean", "length:min", "length:max", "length:variance","length:cv", "length:unique/len(data_list)", "whitespace_ratios:mean","punctuation_ratios:mean","special_character_ratios:mean","numeric_ratios:mean", "whitespace_ratios:cv","punctuation_ratios:cv","special_character_ratios:cv","numeric_ratios:cv", "colname:bleu_score", "colname:edit_distance","colname:lcs","colname:tsm_cosine", "colname:one_in_one", "colname:all_same"

- tsm_cosine: cosine similarity computed by sentence-transformers using "paraphrase-multilingual-mpnet-base-v2". Support multi-language column names matching.
