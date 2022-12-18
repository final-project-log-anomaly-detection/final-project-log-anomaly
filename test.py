from common.utils import seed_everything, dump_final_results, dump_params
from common.dataloader import load_sessions, log_dataset
# from deeploglizer.common.preprocess import FeatureExtractor
# from deeploglizer.models import LSTM
from torch.utils.data import DataLoader
import argparse
import sys
import logging
import json

sys.path.append("../")


parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", default="BGL", type=str)
parser.add_argument(
    "--data_dir",
    default="/Users/thanadonlamsan/Documents/research project จบ/final_project_code/final-project-log-anomaly/data_preprocess/processed/BGL_preprocessed/bgl_0.0_tar",
    type=str,
)

# Model params
parser.add_argument("--model_name", default="LSTM", type=str)
parser.add_argument("--use_attention", action="store_true")
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--num_directions", default=2, type=int)
parser.add_argument("--embedding_dim", default=32, type=int)

# Input params
parser.add_argument("--feature_type", default="semantics", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--label_type", default="next_log", type=str)
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

# Training params
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--topk", default=10, type=int)
parser.add_argument("--patience", default=3, type=int)

# Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)


params = vars(parser.parse_args())

model_save_path = dump_params(params)

if __name__ == '__main__':
    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])

    dataloader_train = DataLoader(session_train, batch_size=params["batch_size"], shuffle=False, pin_memory=True)
    # dataset_test = log_dataset(session_test, feature_type=params["feature_type"])
    # print(dataset_test)
