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
        # self.proj = nn.Linear(768, out_dims)
        # self.weights = nn.Parameter(torch.ones(12))
        # self.proj_1 = nn.Linear(768, 384)
        # self.activation = nn.ReLU()
        # self.proj_2 = nn.Linear(384, out_dims)
        self.lstm = nn.LSTM(768, 128, 1, bidirectional=True)
        self.proj = nn.Linear(128 * 2, out_dims)

    def forward(self, feat):
        hidden = self.model(feat)
        # all_hidden = hidden.hidden_states[1:]
        # intermediate = torch.stack(all_hidden, dim=-1)
        # intermediate = torch.mul(intermediate, torch.nn.functional.softmax(self.weights)).sum(dim=-1)
        # layer_10 = all_hidden[10]
        # output = self.proj(hidden.last_hidden_state)
        # output = self.proj(layer_10)
        # output = self.proj_1(hidden.last_hidden_state)
        # output = self.activation(output)
        # output = self.proj_2(output)
        output,_ = self.lstm(hidden.last_hidden_state)
        output = self.proj(output)
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