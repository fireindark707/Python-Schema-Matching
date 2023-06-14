from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import torch
import random
import math
import pandas as pd
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
now_pth = os.path.dirname(os.path.abspath(__file__))

class HXLTagger():
    def __init__(self):
        self.peft_model_id = "mt0-xl_LORA_SEQ_2_SEQ_LM"
        self.model_pth = os.path.join(now_pth, self.peft_model_id)
        self.model_name_or_path = "bigscience/mt0-xl"
        self.tokenizer_name_or_path = "bigscience/mt0-xl"
        self.load_model_tokenizer()
    
    def load_model_tokenizer(self):
        config = PeftConfig.from_pretrained(self.model_pth)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(model, self.model_pth)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.model.eval()
    
    def parse_tag(self, tag):
        # #tag+attribute
        tagname = tag.split("+")[0].strip("#")
        attributes = tag.strip().split("+")[1:]
        return tagname, attributes

    def generate(self, prompts):
        tags = []
        for i in range(len(prompts)):
            inputs = self.tokenizer(prompts[i], return_tensors="pt")
            prompt = prompts[i]

            with torch.no_grad():
                outputs = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=30)
                predicted = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            
            tags.append(predicted)
        return tags
    
    def predict(self, column_name, examples_list):
        random.shuffle(examples_list)
        examples_list_string = ""
        for i,example in enumerate(examples_list):
            if example == "" or example is math.nan or example is None:
                continue
            examples_list_string += f"'{example}', "
            if len(examples_list_string) > 120 or i >= 9:
                break
        examples_list_string = examples_list_string[:-2]
        prompt = f"Column Name: {column_name} Examples: [{examples_list_string}]"
        tags = self.generate([prompt])
        return tags[0]
    
    def get_hxl_tags(self, df):
        col_len = len(df.columns)
        tags = []
        for i in tqdm(range(col_len)):
            colname = df.columns[i]
            col = df.iloc[:, i]
            col = col.dropna().tolist()
            hxl_tag = self.predict(colname, col)
            tag_name, attributes = self.parse_tag(hxl_tag)
            tags.append(
                {
                "tagname": tag_name,
                "attributes": attributes,
                "full": hxl_tag
                }
            )
        return tags

if __name__ == "__main__":
    randome_tests = [
        "Column Name: 链接 Examples: ['www.google.com', 'www.sohu.com', 'www.yahoo.com']",
        "Column Name: 电影名称 Examples: ['急速时刻', '变形金刚', '阿甘正传']",
        "Column Name: 坪数 Examples: ['5', '3', '6', '8']", 
        "Column Name: Authors Examples: ['Zeisel A.', 'Farlik M.', 'Smallwood S.', 'Blaschke K.']", 
        "Column Name: 國家 Examples: ['USA', 'UK', 'IR', 'AUS', 'SA', 'KR', 'JP']",
        "Column Name: 人名 Examples: ['James Bond', 'Robin Sun', 'kevin Zhang', 'Green David']",
        "Column Name: Color Examples: ['0xfe0302', '0xffffff', '0xffffff']",
        "Column Name: 售價 Examples: ['20.0', '31.0', '499.0']",
        "Column Name: City Examples: ['New york', 'London', 'Shanghai']",
        "Column Name: Total Assets Examples: ['272557.05', '344085.12', '367482.39', '389506.24', '441684.56']",
        "Column Name: Total Non-Current Assets Examples: ['179837.14', '233704.42', '249336.4 ', '265197.07', '272331.12']",
        "Column Name: Total Equity Examples: ['152647.22', '197379.46', '208461.87', '218830.05', '225654.67']",
        "Column Name: Equity Attributable to Parent Stockholders  Examples: ['136348.2 ', '181541.51', '193684.07', '204071.82', '204477.33']",
    ]
    tests2 = [
        {
            "column_name": "链接",
            "examples": ['', None, 'www.yahoo.com']
        },
        {
            "column_name": "电影名称",
            "examples": ['急速时刻', '变形金刚', '阿甘正传', '急速时刻']
        },
        {
            "column_name": "坪数",
            "examples": ['', '3', '6', '8']
        },
        {
            "column_name": "Authors",
            "examples": ['Zeisel A.', 'Farlik M.', 'Smallwood S.', 'Blaschke K.']
        },
        {
            "column_name": "國家",
            "examples": ['USA', 'UK', 'IR', 'AUS', 'SA', 'KR', 'JP']
        }
    ]
    table_pth = "Test Data/authors/Table1.csv"
    hxl_tagger = HXLTagger()
    df = pd.read_csv(table_pth)

    tags = hxl_tagger.get_hxl_tags(df)
    print(tags)

