import os
import json
import csv
from datetime import datetime
import random
import math
import argparse

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


STOP_WORDS = stopwords.words('english')

P_TRAIN = 0.85
P_DEV = 0.075
P_TEST = 0.075

def preprocess_body(body):
    new_body = []
    for para in body:
        new_para = []
        for sent in para:
            new_sent = preprocess_sentence(sent)
            new_para.append(new_sent)
        new_body.append(new_para)
    return new_body

def preprocess_sentence(txt):
    words = word_tokenize(txt)
    words = [word for word in words if (word.isalpha() and (word not in STOP_WORDS) )]
    return " ".join(words)


def load_from_raw_data(nela_path):
    BASE_PATH = nela_path
    column_labels = ['headline', 'body', 'source']
    df = pd.DataFrame(columns=column_labels)
    
    for month_dir in os.listdir(BASE_PATH):
        for date_dir in os.listdir(os.path.join(BASE_PATH, month_dir)):
            for source_dir in os.listdir(os.path.join(BASE_PATH, month_dir, date_dir)):
                for article_filename in os.listdir(os.path.join(BASE_PATH, month_dir, date_dir, source_dir)):
                    with open(os.path.join(BASE_PATH, month_dir, date_dir, source_dir, article_filename)) as f:
                        json_txt = f.read()
                        try:
                            article_dict = json.loads(json_txt)
                        except json.decoder.JSONDecodeError:
                            continue
                        article_dict = {
                            "headline": article_dict["title"],
                            "body": article_dict["content"],
                            "source": article_dict["source"],
                        }
                        df = df.append(article_dict, ignore_index=True)
    
    print(df)
    return df    

def export_csv_for_prediction(df, source_path, para_flag=False):
    df = df.dropna()
    df = df.sample(n=5)
    print(df.iloc[0].body)
    df['body'] = df.apply(lambda row: list(map(lambda x: [x], row['body'].split("\n\n"))), axis=1)
    df['headline'] = df.apply(lambda row: preprocess_sentence(row['headline']), axis=1)
    df['body'] = df.apply(lambda row: preprocess_body(row['body']), axis=1)

    print(df)
    print(df.iloc[0].body)
    f = open(source_path, "w")
    writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in df.itertuples():
        if para_flag:
            for para in row.body:
                writer.writerow([row.Index, row.headline, " ".join(para)])
        else:
            for para in row.body:
                for sent in para:
                    writer.writerow([row.Index, row.headline, sent])


def create_dataset(df):
    df = df.sample(frac=1)
    number_array = [i for i in range(200)]
    label_0_list = []
    label_1_list = []
    para_list = []
    label_1_type = 0
    type_1_avr_len = []
    result_df = []
    result_df_para = []
    for i in range(df.shape[0] // 3):
        label_0_row = df.iloc[3*i]
        label_1_base = df.iloc[3*i + 1]
        label_1_attach = df.iloc[3*i + 2]
        base_paras = label_1_base["paras"]
        attach_paras = label_1_attach["paras"]
        base_para_len = len(label_1_base["paras"])
        attach_para_len = len(label_1_attach["paras"])

        if label_1_type == 0: # Applying rule (1) with only one paragraph
            selected_para_index = random.randrange(0, attach_para_len)
            insert_index = random.randrange(0, base_para_len + 1)
            
            fake_paras = attach_paras[selected_para_index]
            result_article = base_paras.copy()
            result_article.insert(insert_index, attach_paras[selected_para_index])
            fake_paras = [fake_paras]

        elif label_1_type == 1: # Applying rule (1) with two or more consecutive paragraphs
            selected_insert_para_len = random.randrange(1, min([math.ceil(base_para_len / 2), attach_para_len]) + 1)
            type_1_avr_len.append(selected_insert_para_len)
            selected_para_start_index = random.randrange(0, attach_para_len - selected_insert_para_len + 1)
            insert_index = random.randrange(0, base_para_len + 1)
            
            fake_paras = attach_paras[selected_para_start_index:selected_para_start_index + selected_insert_para_len]
            result_article = base_paras[:insert_index] + fake_paras + base_paras[insert_index:]

        elif label_1_type == 2: # Applying rule (2) without random arrangement (maintaining the ordering of the sampled paragraphs)
            selected_insert_para_len = random.randrange(1, min([math.ceil(base_para_len / 2), attach_para_len]) + 1)
            type_1_avr_len.append(selected_insert_para_len)
            selected_para_indices = random.sample(number_array[:attach_para_len], selected_insert_para_len)
            insert_indices = random.sample(number_array[:base_para_len + selected_insert_para_len], selected_insert_para_len)
            fake_paras = [attach_paras[i] for i in selected_para_indices]
            
            base_paras_temp = base_paras.copy()
            fake_paras_temp = fake_paras.copy()
            base_paras_temp.reverse()
            fake_paras_temp.reverse()
            result_article = []
            for i in range(base_para_len + selected_insert_para_len):
                if i in insert_indices:
                    result_article.append(fake_paras_temp.pop())
                else:
                    result_article.append(base_paras_temp.pop())

        elif label_1_type == 3: # Applying rule (2) (n > 1)
            selected_insert_para_len = random.randrange(1, min([math.ceil(base_para_len / 2), attach_para_len]) + 1)
            type_1_avr_len.append(selected_insert_para_len)
            selected_para_indices = random.sample(number_array[:attach_para_len], selected_insert_para_len)
            insert_indices = random.sample(number_array[:base_para_len + selected_insert_para_len], selected_insert_para_len)
            fake_paras = [attach_paras[i] for i in selected_para_indices]

            base_paras_temp = base_paras.copy()
            fake_paras_temp = fake_paras.copy()
            result_article = base_paras[:]
            for fake_para in fake_paras:
                result_article.insert(random.randrange(0, len(result_article)), fake_para)

        label_1_type = (label_1_type + 1) % 4
        
        result_df.append({"headline": label_0_row["headline"], "body": label_0_row["paras"], "fake_type": -1, "label": 0})
        result_df.append({"headline": label_1_base["headline"], "body": result_article, "fake_type": label_1_type, "label": 1})
        result_df_para.append({"headline": label_0_row["headline"], "body": label_0_row["paras"], "fake_type": -1, "label": 0, "fake_paras": [], "base": label_0_row["paras"]})
        result_df_para.append({"headline": label_1_base["headline"], "body": result_article, "fake_type": label_1_type, "label": 1, "fake_paras": fake_paras, "base": base_paras})

    result_df = pd.DataFrame(result_df)
    result_df_para = pd.DataFrame(result_df_para)
    return result_df, result_df_para


def export_df_to_dataset(df, output_dir):
    df["body"] = df.apply(lambda row: [sent_tokenize(para) for para in row["body"]], axis=1)

    def tokenize_sent_in_para(para):
        result = []
        for sent in para:
            result.append(" ".join(word_tokenize(sent)))
        result = " <EOS> ".join(result) + " <EOS> "
        return result
    df["body"] = df.apply(lambda row: [tokenize_sent_in_para(para) for para in row["body"]], axis=1)

    df["headline"] = df.apply(lambda row: " ".join(word_tokenize(row["headline"])), axis=1)
    df["body_merged"] = df.apply(lambda row: " <EOP> ".join(row["body"]).replace("<EOS>  <EOP>", "<EOS> <EOP>"), axis=1)

    df_train = df.iloc[:int(df.shape[0] * P_TRAIN)]
    df_dev = df.iloc[int(df.shape[0] * P_TRAIN):int(df.shape[0] * (P_TRAIN + P_DEV))]
    df_test = df.iloc[int(df.shape[0] * (P_TRAIN + P_DEV)):]
    
    whole_train_df = df_train[["headline", "body_merged", "label"]]
    whole_train_df.to_csv(os.path.join(output_dir, "train.csv"), encoding="utf-8", header=False)
    whole_dev_df = df_dev[["headline", "body_merged", "label"]]
    whole_dev_df.to_csv(os.path.join(output_dir,"dev.csv"), encoding="utf-8", header=False)
    whole_test_df = df_test[["headline", "body_merged", "label"]]
    whole_test_df.to_csv(os.path.join(output_dir,"test.csv"), encoding="utf-8", header=False)


def export_df_to_dataset_para(df, output_dir):
    df["fake_paras"] = df.apply(lambda row: [sent_tokenize(para) for para in row["fake_paras"]], axis=1)
    df["base"] = df.apply(lambda row: [sent_tokenize(para) for para in row["base"]], axis=1)

    def tokenize_sent_in_para(para):
        result = []
        for sent in para:
            result.append(" ".join(word_tokenize(sent)))
        result = " <EOS> ".join(result) + " <EOS> "
        return result
    df["fake_paras"] = df.apply(lambda row: [tokenize_sent_in_para(para) for para in row["fake_paras"]], axis=1)
    df["base"] = df.apply(lambda row: [tokenize_sent_in_para(para) for para in row["base"]], axis=1)

    df["headline"] = df.apply(lambda row: " ".join(word_tokenize(row["headline"])), axis=1)

    df_train = df.iloc[:int(df.shape[0] * P_TRAIN)]
    df_dev = df.iloc[int(df.shape[0] * P_TRAIN):int(df.shape[0] * (P_TRAIN + P_DEV))]
    df_test = df.iloc[int(df.shape[0] * (P_TRAIN + P_DEV)):]

    para_train_df = []
    for row in df_train.itertuples():
        for para in row.fake_paras:
            para_train_df.append([row.Index, row.headline, para, 1])
        for para in row.base:
            para_train_df.append([row.Index, row.headline, para, 0])
    para_dev_df = []
    for row in df_dev.itertuples():
        for para in row.fake_paras:
            para_dev_df.append([row.Index, row.headline, para, 1])
        for para in row.base:
            para_dev_df.append([row.Index, row.headline, para, 0])
    
    with open(os.path.join(output_dir,"train_IP.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in para_train_df:
            writer.writerow(row)

    with open(os.path.join(output_dir,"dev_IP.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in para_dev_df:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help="path to article csv file")
    parser.add_argument('--output_dir', default=".", help="directory to export output file")
    parser.add_argument('--nela_path', default=None, help="directory to raw NELA JSON folder path")


    args = parser.parse_args()
    if args.nela_path is not None:
        df = load_from_raw_data(args.nela_path)
    else:
        df = pd.read_csv(args.input_path, header=None, names=["headline", "body"])
    
    df["body"] = df.apply(lambda row: row["body"].replace("\xa0", " "), axis=1)
    df["paras"] = df.apply(lambda row: list(filter(None, row["body"].split("\n"))), axis=1)
    print("Shuffling....")
    df, df_para = create_dataset(df)
    print("Exporting....")
    export_df_to_dataset(df, args.output_dir)
    export_df_to_dataset_para(df_para, args.output_dir)
