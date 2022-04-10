# Schema Matching by XGboost
Using XGboost to perform schema matching task on tables. Support multi-language column names matching and can be used without column names.

## What is schema matching?

![](https://media.springernature.com/original/springer-static/image/prt%3A978-3-319-77525-8%2F19/MediaObjects/978-3-319-77525-8_19_Part_Fig1-20_HTML.png)

Schema matching is the problem of finding potential associations between elements (most often attributes or relations) of two schemas. [https://link.springer.com/referenceworkentry/10.1007/978-3-319-77525-8_20]

## Dependencies

- numpy==1.19.5
- pandas==1.1.5
- nltk==3.6.5
- python-dateutil==2.8.2
- sentence-transformers==2.1.0
- xgboost==1.5.2
- strsimpy==0.2.1

## Data

See Data format in Training Data and Test Data folders. You need to put mapping.txt, Table1.csv and Table2.csv in new folders under Training Data. For Test Data, mapping.txt is not needed.

## Output

- similarity_matrix_label.csv: Labels(0,1) for each column pairs.
- similarity_matrix_value.csv: Average of raw values computed by all the xgboost models.

## Usage

### 1.Construct features
```
python relation_features.py
```
### 2.Train xgboost models
```
python train.py
```
### 3.Calculate similarity matrix (inference)
```
Example: 
python cal_column_similarity.py -p Test\ Data/self -m model/2022-04-10-20-48-57
python cal_column_similarity.py -p Test\ Data/authors -m model/2022-04-10-21-39-11 -t 0.9
```
Parameters:
- -p: Path to test data folder, must contain "Table1.csv" and "Table2.csv"
- -m: Path to trained model folder, which must contain at least one pair of ".model" file and ".threshold" file.
- -t: Threshold, you can use this parameter to specify threshold value, suggest 0.9 for easy matching(column name very similar). Default value is calculated from training data, which is around 0.15-0.2. This value is used for difficult matching(column name masked or very different).
## Feature Engineering

Features: "is_url","is_numeric","is_date","is_string","numeric:mean", "numeric:min", "numeric:max", "numeric:variance","numeric:cv", "numeric:unique/len(data_list)", "length:mean", "length:min", "length:max", "length:variance","length:cv", "length:unique/len(data_list)", "whitespace_ratios:mean","punctuation_ratios:mean","special_character_ratios:mean","numeric_ratios:mean", "whitespace_ratios:cv","punctuation_ratios:cv","special_character_ratios:cv","numeric_ratios:cv", "colname:bleu_score", "colname:edit_distance","colname:lcs","colname:tsm_cosine", "colname:one_in_one"

- tsm_cosine: cosine similarity computed by sentence-transformers using "paraphrase-multilingual-mpnet-base-v2". Support multi-language column names matching.

## Performance

### Cross Validation on Training Data(Each pair to be used as test data)

- Average Precision: 0.70
- Average Recall: 0.82
- Average F1: 0.73

Average Confusion Matrix:
|                | Negative(Truth) | Positive(Truth) |
|----------------|-----------------|-----------------|
| Negative(pred) | 0.92109479      | 0.07890521      |
| Positive(pred) | 0.1765625       | 0.8234375       |

### Inference on Test Data (Give confusing column names)

Data: https://github.com/fireindark707/Schema_Matching_XGboost/tree/main/Test%20Data/self

|         | title      | text       | summary    | keywords   | url        | country    | language   | domain     | name  | timestamp  |
|---------|------------|------------|------------|------------|------------|------------|------------|------------|-------|------------|
| col1    | 1(correct) | 0          | 0          | 0          | 0          | 0          | 0          | 0          | 0     | 0          |
| col2    | 0          | 1(correct) | 0          | 0          | 0          | 0          | 0          | 0          | 0     | 0          |
| col3    | 0          | 0          | 1(correct) | 0          | 0          | 0          | 0          | 0          | 0     | 0          |
| words   | 0          | 0          | 0          | 1(correct) | 0          | 0          | 0          | 0          | 0     | 0          |
| link    | 0          | 0          | 0          | 0          | 1(correct) | 0          | 0          | 0          | 0     | 0          |
| col6    | 0          | 0          | 0          | 0          | 0          | 1(correct) | 0          | 0          | 0     | 0          |
| lang    | 0          | 0          | 0          | 0          | 0          | 0          | 1(correct) | 0          | 0     | 0          |
| col8    | 0          | 0          | 0          | 0          | 0          | 0          | 0          | 1(correct) | 0     | 0          |
| website | 0          | 0          | 0          | 0          | 0          | 1(FP)      | 0          | 0          | 0(FN) | 0          |
| col10   | 0          | 0          | 0          | 0          | 1(FP)      | 0          | 0          | 0          | 0     | 1(correct) |

## Cite
```
@software{fireinfark707_Schema_Matching_by_2022,  
author = {fireinfark707},  
license = {MIT},  
month = {4},  
title = {{Schema Matching by XGboost}},  
url = {https://github.com/fireindark707/Schema_Matching_XGboost},  
year = {2022}  
}
```
