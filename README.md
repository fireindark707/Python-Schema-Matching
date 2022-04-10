# Schema Matching by XGboost
Using XGboost to perform schema matching task on tables.

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
