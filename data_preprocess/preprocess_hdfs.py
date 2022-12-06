import numpy as np
import argparse
import pandas as pd

import os
import re
from collections import OrderedDict, defaultdict
import pickle
from utils import decision, json_pretty_dump

seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.0, type=float)

params = vars(parser.parse_args())

data_name = f'hdfs_{params["train_anomaly_ratio"]}_tar'
data_dir = "data_preprocess/processed/HDFS"

params = {
    "log_file":  "Tokenize_HDFS/HDFS_2_tokenize.csv",
    "label_file": "HDFS_1/anomaly_label.csv",
    "test_ratio": 0.2,
    "random_sessions": True,  # shuffle sessions
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}


data_dir = os.path.join(data_dir, data_name)
os.makedirs(data_dir, exist_ok=True)


def preprocess_hdfs(log_file, label_file, test_ratio=None, train_anomaly_ratio=1, random_sessions=False, **kwargs):
    print("Loading HDFS logs from {}.".format(log_file))

    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    label_data = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)
    label_data["Label"] = label_data["Label"].map(lambda x: int(x == "Anomaly"))  # anomaly == 1
    label_data_dict = dict(zip(label_data["BlockId"], label_data["Label"]))  # zip label data from csv to dict

    session_dict = OrderedDict()  # remember insertion member
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}  # split index, data

    for _, row in enumerate(struct_log.values):
        blkId_list = re.findall(r"(blk_-?\d+)", row[column_idx["Content"]])  # list block_id ออกมา
        blkId_set = set(blkId_list)  # change block_id to set
        for blk_Id in blkId_set:
            if blk_Id not in session_dict:  # ถ้าไม่มีใน session_dict ให้สร้าง defaultdict value equal list inside key equal block_id
                session_dict[blk_Id] = defaultdict(list)
            # add template that match block_id
            session_dict[blk_Id]["templates"].append(row[column_idx["EventTemplate"]])
            # {'block_id':{'templates': ['BLOCK* NameSystem.allocateBlock:<*>',Receiving block <*> src: /<*> dest: /<*>]}}

    for k in session_dict.keys():
        session_dict[k]["label"] = label_data_dict[k]  # ใส่ label 0,1 ใน orderDict

    session_idx = list(range(len(session_dict)))  # array ขนาดเท่ากับ block_id ที่ใส่ตัวเลขเข้าไปเป็นลำดับด้วย

    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)  # สลับตำแหน่งใน list

    session_ids = np.array(list(session_dict.keys()))  # block_id มาทำเป็น array
    session_labels = np.array(list(map(lambda x: label_data_dict[x], session_ids)))  # map label ออกมาเป็น list 0,1

    number_train_lines = int((1 - test_ratio) * len(session_idx))  # จะเอามา train กี่ line
    number_test_lines = int(test_ratio * len(session_idx))  # จะเอามา test กี่ line

    # แบ่ง id ที่จะใช้ train โดย list ที่ผ่านการสลับ ตน. ของตัวเลขแล้ว
    session_idx_train = session_idx[0:number_train_lines]
    session_idx_test = session_idx[-number_test_lines:]

    session_id_train = session_ids[session_idx_train]  # list ของ block_id มา map เพื่อเอาข้อมูลแบบสุ่มมา train
    session_id_test = session_ids[session_idx_test]

    session_labels_train = session_labels[session_idx_train]  # ตอนนี้ มี anomaly ปนอยู่ anomaly == 1
    session_labels_test = session_labels[session_idx_test]  # ตอนนี้ มี anomaly ปนอยู่ anomaly == 1

    print("Total # sessions: {}".format(len(session_ids)))  # number of block_id

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (session_dict[k]["label"] == 0) or (session_dict[k]["label"] == 1 and decision(train_anomaly_ratio))
    }
    print("have false i train session: ", 1 in session_train.values())

    session_test = {k: session_dict[k] for k in session_id_test}  # test ไม่ต้องมีเงื่อนไข

    session_labels_train = [v["label"] for k, v in session_train.items()]
    session_labels_test = [v["label"] for k, v in session_test.items()]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))

    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    preprocess_hdfs(**params)
