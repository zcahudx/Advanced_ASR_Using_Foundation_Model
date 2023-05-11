import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio


class TDataset(Dataset):
    def __init__(self, path, vocab):
        with open(path) as f:
            self.data = json.load(f)
        self.data_idx = list(self.data.keys())
        # vocab = {}
        # with open(vocab_path) as f:
        #     for id, text in enumerate(f): 
        #         vocab[text.strip()] = id
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[self.data_idx[index]]
        data_path = data["fbank"]
        fbank = torch.load(data_path)
        fbank_mean = torch.mean(fbank, dim=0, keepdims=True)
        fbank_std = torch.std(fbank, dim=0, keepdims=True)
        fbank = (fbank - fbank_mean) / fbank_std
        phn = data["phn"]
        duration = data["duration"]
        return fbank, phn, duration


class TDatasetwav(TDataset):
    def __getitem__(self, index):
        data = self.data[self.data_idx[index]]
        data_path = data["wav"]
        wav, _ = torchaudio.load(data_path)
        wav = wav[0]
        wav_mean = torch.mean(wav, dim=0, keepdims=True)
        wav_std = torch.std(wav, dim=0, keepdims=True)
        wav = (wav - wav_mean) / wav_std
        word = data["word"]
        tgt_list = []
        for i in word:
            if i in self.vocab.keys():
                tgt_list.append(self.vocab[i])
            elif i == " ":
                tgt_list.append(self.vocab["<space>"])
            else:
                tgt_list.append(self.vocab["<unk>"])
        
        tgt = torch.tensor(tgt_list)
        duration = data["duration"]
        return wav, tgt, duration


def collate_wrapper(batch):
    fbank = pad_sequence([i[0] for i in batch])
    lens = torch.tensor([len(i[0]) for i in batch], dtype=torch.long)
    tgt = [i[1] for i in batch]
    duration = [i[2] for i in batch]
    return fbank, lens, tgt, duration


def get_dataloader(path, vocab, bs, shuffle):
    dataset = TDataset(path, vocab)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        pin_memory=True,
    )

def get_dataloader_wav(path, vocab, bs, shuffle):
    dataset = TDatasetwav(path, vocab)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        pin_memory=True,
    )

