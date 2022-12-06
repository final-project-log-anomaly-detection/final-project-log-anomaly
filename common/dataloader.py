import os
import pickle
import json
import logging


def load_sessions(data_dir):
    with open(os.path.join(data_dir, "data_desc.json"), "r") as fr:
        data_desc = json.load(fr)
    with open(os.path.join(data_dir, "session_train.pkl"), "rb") as fr:
        session_train = pickle.load(fr)
    with open(os.path.join(data_dir, "session_test.pkl"), "rb") as fr:
        session_test = pickle.load(fr)

    train_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0) for _, v in session_train.items()
    ]

    test_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0) for _, v in session_test.items()
    ]

    num_train = len(session_train)
    ratio_train = sum(train_labels) / num_train
    num_test = len(session_test)
    ratio_test = sum(test_labels) / num_test
    # logging.info("Load from {}".format(data_dir))
    # logging.info(json.dumps(data_desc, indent=4))
    logging.info("# train sessions {} ({:.2f} anomalies)".format(num_train, ratio_train))
    logging.info("# test sessions {} ({:.2f} anomalies)".format(num_test, ratio_test))
    return session_train, session_test
