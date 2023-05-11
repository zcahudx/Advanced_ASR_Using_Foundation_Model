from dataloader import get_dataloader
import torch
from collections import Counter
from datetime import datetime
from trainer import train
import models
from decoder import decode
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = 'Running MLMI2 experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--train_json', type=str, default="data/train_10h.json")
parser.add_argument('--val_json', type=str, default="data/devsubset.json")
parser.add_argument('--test_clean_json', type=str, default="data/test_clean.json")
parser.add_argument('--test_other_json', type=str, default="data/test_other.json")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=1, help="number of rnn layers")
parser.add_argument('--fbank_dims', type=int, default=23, help="filterbank dimension")
parser.add_argument('--model_dims', type=int, default=128, help="model size for rnn layers")
parser.add_argument('--concat', type=int, default=3, help="concatenating frames")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--vocab', type=str, default="data/vocab.txt", help="vocabulary file path")
parser.add_argument('--use_fbank', action="store_true")
parser.add_argument('--model', type=str, default="wav2vec2")
parser.add_argument('--report_interval', type=int, default=50, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=10)
args = parser.parse_args()


vocab = {}
with open(args.vocab) as f:
    for id, text in enumerate(f):
        vocab[text.strip()] = id

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(args)
args.device = device
args.vocab = vocab

if args.model == "wav2vec2":
    model = models.Wav2Vec2CTC(len(args.vocab))
else:
    model = models.BiLSTM(args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(args.vocab))

num_params = sum(p.numel() for p in model.parameters())
print('Total number of model parameters is {}'.format(num_params))

model.to(args.device)
model_path = "/home/yl879/rds/hpc-work/MLMI14/exp_librispeech/checkpoints/20230306_130307/model_9"

print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
model.to(device)
results_clean = decode(model, args, args.test_clean_json)
print("Results for test_clean:")
print("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, WER: {:.2f}%".format(*results_clean))
results_other = decode(model, args, args.test_other_json)
print("Results for test_other:")
print("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, WER: {:.2f}%".format(*results_other))
