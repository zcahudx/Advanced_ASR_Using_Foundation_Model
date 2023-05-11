import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, WavLMModel

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output


class Wav2Vec2CTC(nn.Module):

    def __init__(self, out_dims):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.freeze_feature_encoder()
        self.proj = nn.Linear(768, out_dim)

    def forward(self, feat):
        hidden = self.model(feat)
        output = self.proj(hidden.last_hidden_state)
        return output

class WavLMCTC(nn.Module):

    def __init__(self, out_dims):
        super().__init__()
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.model.freeze_feature_encoder()
        self.proj = nn.Linear(768, out_dims)
        
    def forward(self, feat):
        hidden = self.model(feat)
        output = self.proj(hidden.last_hidden_state)
        return output
