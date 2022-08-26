import os, json, re
import torch
from data import DataModule
import pytorch_lightning as pl
import torchvision


class FeatureExtractor(torch.nn.Module):

    def __init__(self, input_size=(64, 320), output_len=20):
        """
        Model for feature extraction.
        Args:
          - input_size: Input size
          - output_len: Output length
        """
        super(FeatureExtractor, self).__init__()
        h, w = input_size
        resnet = torchvision.models.resnet18(weights=True)
        self.cnn = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.pool = torch.nn.AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = torch.nn.Conv2d(w // 32, output_len, kernel_size=1)
        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    def apply_projection(self, x):
        """
        Uses convolution to increase width of a features.
        Args:
            - x: Tensor of features (shaped B x C x H x W).
        Returns:
            New tensor of features (shaped B x C x H x W').
        """
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x):
        features = self.cnn(x) # conv layers
        features = self.pool(features) # to make height == 1
        features = self.apply_projection(features) # to increase width
        return features


class SequencePredictor(torch.nn.Module):

    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False
    ):
        """
        Model which predict sequence of text on a car plate.
        Args:
          - input_size: Input size
          - hidden_size: Hidden size
          - num_layers: Number of layers
          - num_classes: Number of classes
          - dropout: Dropout
          - bidirectional: If True model will be bidirectional, otherwise - unidirectional. 
        """
        super(SequencePredictor, self).__init__()
        self.num_classes = num_classes
        self.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = torch.nn.Linear(in_features=fc_in, out_features=num_classes)

    def _init_hidden(self, batch_size):
        """
        Initialize new tensor of zeroes for RNN hidden state.
        Args:
            - batch_size: Int size of batch
        Returns:
            Tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        num_directions = 2 if self.rnn.bidirectional else 1
        h = torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        return h

    def _reshape_features(self, x):
        """
        Change dimensions of x to fit RNN expected input.
        Args:
            - x: Tensor x shaped (B x (C=1) x H x W).
        Returns:
            New tensor shaped (W x B x H).
        """
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        return x

    def forward(self, x):
        x = self._reshape_features(x)
        batch_size = x.size(1)
        h_0 = self._init_hidden(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)
        x = self.fc(x)
        return x


class CRNN(pl.LightningModule):
    ABC = "0123456789ABCEHKMOPTXY"
    
    def __init__(
        self, 
        alphabet=ABC,
        cnn_input_size=(64, 320),
        cnn_output_len=20,
        rnn_hidden_size=128,
        rnn_num_layers=1,
        rnn_dropout=0.0,
        rnn_bidirectional=False,
        lr=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(CRNN, self).__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(
            input_size=cnn_input_size,
            output_len=cnn_output_len
        )
        self.sequence_predictor = SequencePredictor(
            input_size=self.features_extractor.num_output_features,
            hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
            num_classes=len(alphabet) + 1, dropout=rnn_dropout,
            bidirectional=rnn_bidirectional
        )
        self.lr = lr
    
    @staticmethod
    def decode_sequence(pred, abc):
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        outputs = []
        for i in range(len(pred)):
            outputs.append(pred_to_string(pred[i], abc))
        return outputs

    def forward(self, x, decode=False):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        if decode:
            sequence = self.decode_sequence(sequence, self.alphabet)
        return sequence
    
    def training_step(self, batch, batch_idx):
        images = batch["images"].to(self.device)
        seqs = batch["seqs"]
        seq_lens = batch["seq_lens"]
        seqs_pred = model(images).cpu()
        log_probs = torch.nn.functional.log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = torch.nn.functional.ctc_loss(
            log_probs, seqs, seq_lens_pred, seq_lens
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(model.parameters(), lr=self.lr)
