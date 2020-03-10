import torch
from torch import nn


class Question_pipeline(nn.Module):
    def __init__(self, config, nb_words, dropout_drop_prob=0, sequence_padding_idx=0):
        super(Question_pipeline, self).__init__()

        # Question Pipeline
        self.word_emb = nn.Embedding(num_embeddings=nb_words,
                                     embedding_dim=config['question']['word_embedding_dim'],
                                     padding_idx=sequence_padding_idx)

        # FIXME : Bidirectional ?
        # FIXME : Are we missing normalization here ?
        self.rnn_state = nn.GRU(input_size=config['question']['word_embedding_dim'],
                                hidden_size=config["question"]["rnn_state_size"],
                                batch_first=True,
                                dropout=0)

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, question, question_lengths, pack_sequence):
        word_emb = self.word_emb(question)
        word_emb = self.dropout(word_emb)

        if pack_sequence:
            word_emb = torch.nn.utils.rnn.pack_padded_sequence(word_emb, question_lengths, batch_first=True,
                                                               enforce_sorted=False)

        rnn_out, rnn_hidden = self.rnn_state(word_emb)

        rnn_hidden_state = self.dropout(rnn_hidden.squeeze(0))

        return rnn_hidden_state