import os
import pickle
import json
import logging
from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)


def load_sessions(data_dir):
    with open(os.path.join(data_dir, "data_desc.json"), "r") as fr:
        data_desc = json.load(fr)
    with open(os.path.join(data_dir, "session_train.pkl"), "rb") as fr:
        session_train = pickle.load(fr)
    with open(os.path.join(data_dir, "session_test.pkl"), "rb") as fr:
        session_test = pickle.load(fr)

    train_labels = [
        v["Label"] if not isinstance(v["Label"], list) else int(sum(v["Label"]) > 0) for _, v in session_train.iterrows()
    ]

    test_labels = [
        v["Label"] if not isinstance(v["Label"], list) else int(sum(v["Label"]) > 0) for _, v in session_test.iterrows()
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


class log_dataset(Dataset):
    def __init__(self, session_dict, feature_type="semantics"):
        self.flatten_data_list = []
        # flatten all sessions
        for session_idx, data_dict in enumerate(session_dict.values()):
            features = data_dict["features"][feature_type]
            _logger.warning(data_dict["features"][feature_type])
            window_labels = data_dict["window_labels"]
            # _logger.warning(data_dict.keys())
            # _logger.warning("windows: {}".format(data_dict["windows"]))
            # _logger.warning("window_labels: {}".format(data_dict["window_labels"]))
            # _logger.warning("window_anomalies: {}".format(data_dict["window_anomalies"]))
            # _logger.warning("features: {}".format(data_dict["features"][feature_type]))
            window_anomalies = data_dict["window_anomalies"]
            for window_idx in range(len(window_labels)):
                sample = {
                    "session_idx": session_idx,  # not session id
                    "features": features[window_idx],
                    "window_labels": window_labels[window_idx],
                    "window_anomalies": window_anomalies[window_idx],
                }
                self.flatten_data_list.append(sample)
        # self.flatten_data_list = flatten_data_list
        # _logger.warning(self.flatten_data_list[0])

    def __len__(self):
        return len(self.flatten_data_list)

    def __getitem__(self, idx):
        return self.flatten_data_list[idx]
