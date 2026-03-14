import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, out_channels, kernel_size=3):
        super().__init__()

        self.char_embedding = nn.Embedding(
            char_vocab_size,
            char_embed_dim,
            padding_idx=0
        )

        self.conv = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, char_ids):
        """
        char_ids: (B, T, W)
        B = batch size
        T = number of words
        W = max characters per word
        """

        B, T, W = char_ids.shape

        # Flatten words so CNN runs on each word independently
        char_ids = char_ids.view(B * T, W)

        # Character embeddings
        chars = self.char_embedding(char_ids) 
        chars = chars.transpose(1, 2)             
        conv_out = self.conv(chars)           
        conv_out = F.relu(conv_out)

        # Max pooling over characters
        word_vectors = torch.max(conv_out, dim=2).values  

        # Reshape back to sentence structure
        word_vectors = word_vectors.view(B, T, -1)    
        return word_vectors

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, char_vocab_size, tag_to_ix, embedding_dim, hidden_dim, char_embed_dim, char_out_channels, run=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.dropout = nn.Dropout(p=config.DROPOUT)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        if run == "CNN":
            self.char_cnn = CharCNN(char_vocab_size, char_embed_dim, char_out_channels)
            bilstm_input_dim = embedding_dim + char_out_channels
        else:
            self.char_cnn = None
            bilstm_input_dim = embedding_dim

        self.lstm = nn.LSTM(bilstm_input_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[config.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[config.STOP_TAG]] = -10000


    def _forward_alg(self, feats, lengths, gold_tags=None, cost_factor=None):
        """
        Compute log partition function Z(x) for entire batch using forward algorithm.
        Args:
            feats  : (B, T, tagset_size)  emission scores
            lengths: (B,)                 actual sequence lengths
        Returns:
            alpha: (B,) log partition function for each sequence
        """
        batch_size, seq_len, num_tags = feats.shape
        device = feats.device

        # Initialize forward variables
        forward_var = torch.full((batch_size, num_tags), -10000., device=device)
        forward_var[:, self.tag_to_ix[config.START_TAG]] = 0.

        tag_range = torch.arange(num_tags, device=device) # for softmax margin

        # Iterate over sequence 
        for t in range(seq_len):
            feat_t = feats[:, t]  # (B, S)

            # Add cost if softmax margin is used
            cost_t = torch.zeros(batch_size, num_tags, device=device)
            if gold_tags is not None and cost_factor is not None:
                gold_t = gold_tags[:, t].unsqueeze(1)               # (B, 1)
                cost_t = cost_factor * (tag_range.unsqueeze(0) != gold_t).float()  # (B, S)

            next_forward = []

            # Compute score for each possible next tag
            for next_tag in range(num_tags):
                # Transition scores
                trans_score = self.transitions[next_tag].unsqueeze(0)  # (1, S)
                # Emission score for next_tag
                emit_score = feat_t[:, next_tag].unsqueeze(1) + cost_t[:, next_tag].unsqueeze(1)  # (B, 1)
                # Score for all possible previous tags
                score = forward_var + trans_score + emit_score  # (B, S)
                next_forward.append(torch.logsumexp(score, dim=1))  # (B)

            # Combine scores for all tags
            next_forward = torch.stack(next_forward, dim=1)  # (B, S)

            # Mask padded tokens
            mask = (t < lengths).float().unsqueeze(1)
            forward_var = mask * next_forward + (1 - mask) * forward_var

        # Transition to STOP
        terminal = forward_var + self.transitions[self.tag_to_ix[config.STOP_TAG]]
        alpha = torch.logsumexp(terminal, dim=1) 
        return alpha

    def _get_lstm_features(self, word_ids, char_ids, lengths):

        batch_size, seq_len = word_ids.shape

        # Word embeddings
        word_emb = self.word_embeds(word_ids)    
        word_emb = self.dropout(word_emb)

        # Character CNN features
        if self.char_cnn is not None:
            char_feats = self.char_cnn(char_ids)    
            embeddings = torch.cat([word_emb, char_feats], dim=2)
        else:
            embeddings = word_emb

        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
        lstm_out = self.dropout(lstm_out)
        feats = self.hidden2tag(lstm_out) 

        return feats

    def _score_sentence(self, feats, tags, lengths):
        """
        Compute the CRF score for the gold tag sequences in a batch.

        Args:
            feats:    (B, T, S) emission scores from the BiLSTM
            tags:     (B, T)    gold tag indices
            lengths:  (B,)      true sentence lengths

        Returns:
            scores:   (B,) score for each gold sequence
        """

        batch_size, max_len, _ = feats.shape
        device = feats.device

        scores = torch.zeros(batch_size, device=device)

        # Add START tag at the beginning of each tag sequence
        start_tags = torch.full(
            (batch_size, 1),
            self.tag_to_ix[config.START_TAG],
            dtype=torch.long,
            device=device
        )

        extended_tags = torch.cat([start_tags, tags], dim=1)

        for t in range(max_len):

            emission_scores = feats[:, t]          # (B, S)

            current_tags = extended_tags[:, t + 1]
            previous_tags = extended_tags[:, t]

            # Transition score: prev_tag → current_tag
            transition_scores = self.transitions[current_tags, previous_tags]

            # Emission score for the correct tag
            emission_scores_for_gold = emission_scores[torch.arange(batch_size), current_tags]

            # Only include scores for valid tokens (not padding)
            valid_mask = (t < lengths).float()

            scores += valid_mask * (transition_scores + emission_scores_for_gold)

        # Add last transition
        last_positions = lengths - 1
        last_tags = tags[torch.arange(batch_size), last_positions]

        scores += self.transitions[self.tag_to_ix[config.STOP_TAG], last_tags]

        return scores

    def _viterbi_decode(self, feats):
        """
        Perform Viterbi decoding for a single sentence.

        Args:
            feats: (T, S) emission scores from the model

        Returns:
            path_score: score of the best tag sequence
            best_path:  list of predicted tag indices
        """

        seq_len, num_tags = feats.shape
        device = feats.device

        # Initialize Viterbi scores
        viterbi_scores = torch.full((num_tags,), -10000., device=device)
        viterbi_scores[self.tag_to_ix[config.START_TAG]] = 0.

        # Backpointer table
        backpointers = torch.zeros(seq_len, num_tags, dtype=torch.long, device=device)

        # Dynamic programming over the sequence
        for t in range(seq_len):

            emission_scores = feats[t]  # (S,)

            # Compute scores for all transitions
            # score(next_tag, prev_tag) = previous_score + transition_score
            transition_scores = viterbi_scores.unsqueeze(0) + self.transitions  # (S, S)

            # Find best previous tag for each current tag
            best_prev_tags = transition_scores.argmax(dim=1)
            best_prev_scores = transition_scores.max(dim=1).values

            # Update Viterbi scores
            viterbi_scores = best_prev_scores + emission_scores

            # Store backpointers
            backpointers[t] = best_prev_tags

        # Add transition to STOP tag
        stop_scores = viterbi_scores + self.transitions[self.tag_to_ix[config.STOP_TAG]]

        best_last_tag = int(stop_scores.argmax().item())
        best_score = stop_scores[best_last_tag]

        # Traceback to recover best path
        best_path = [best_last_tag]

        for t in range(seq_len - 1, -1, -1):
            best_last_tag = int(backpointers[t, best_last_tag].item())
            best_path.append(best_last_tag)

        # Remove START tag
        start_tag = best_path.pop()
        assert start_tag == self.tag_to_ix[config.START_TAG]

        best_path.reverse()

        return best_score, best_path

    def neg_log_likelihood(self, word_ids, char_ids, tags, lengths):
        feats = self._get_lstm_features(word_ids, char_ids, lengths)
        forward_score = self._forward_alg(feats, lengths)
        gold_score = self._score_sentence(feats, tags, lengths)
        loss = (forward_score - gold_score).mean()
        return loss

    def softmax_margin(self, word_ids, char_ids, tags, lengths, cost_factor):
        feats = self._get_lstm_features(word_ids, char_ids, lengths)
        forward_score = self._forward_alg(feats, lengths, tags, cost_factor)
        gold_score = self._score_sentence(feats, tags, lengths)
        loss = (forward_score - gold_score).mean()
        return loss

    def forward(self, word_ids, char_ids, lengths):
        feats = self._get_lstm_features(word_ids, char_ids, lengths)
        best_paths = []
        for b in range(feats.size(0)):
            seq_len = lengths[b].item()
            _, path = self._viterbi_decode(feats[b, :seq_len, :])
            best_paths.append(path)
        return best_paths