import torch
import json
import config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

def load_data(filepath):
    """Load JSON-lines data file; return list of (tokens, tag_ids) tuples."""
    data = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                data.append((d['tokens'], d['tags']))
    return data


def build_vocab(data):
    """Build word and char vocabularies from training data."""
    word_counts = {}
    char_counts = {}
    word_to_idx = {config.PAD_WORD: 0, config.UNK_WORD: 1}
    char_to_idx = {config.PAD_CHAR: 0, config.UNK_CHAR: 1}

    for tokens, _ in data:
        for w in tokens:
            word_counts[w] = word_counts.get(w, 0) + 1
            for c in w:
                char_counts[c] = char_counts.get(c, 0) + 1

    for w in word_counts:
        word_to_idx[w] = len(word_to_idx)

    for c in char_counts:
        char_to_idx[c] = len(char_to_idx)

    return word_to_idx, char_to_idx


def build_tag_vocab():
    """Build tag vocabulary."""
    tag_to_idx = {t: i for i, t in enumerate(config.LABEL_NAMES)}
    tag_to_idx[config.START_TAG] = len(tag_to_idx)
    tag_to_idx[config.STOP_TAG]  = len(tag_to_idx)
    ix_to_tag = {i: t for t, i in tag_to_idx.items()}
    return tag_to_idx, ix_to_tag

class NERDataset(Dataset):
    """Dataset for NER task; converts raw tokens/tags to index tensors."""
    def __init__(self, data, word_to_idx, char_to_idx, tag_to_idx, max_word_len=30):
        self.samples = []
        for tokens, tags in data:
            word_ids = torch.tensor(
                [word_to_idx.get(w, word_to_idx[config.UNK_WORD]) for w in tokens],
                dtype=torch.long)
            # Character IDs: list of tensors, one per word
            char_ids = []
            for w in tokens:
                cids = [char_to_idx.get(c, char_to_idx[config.UNK_CHAR])
                        for c in w[:max_word_len]]
                char_ids.append(torch.tensor(cids, dtype=torch.long))
            tag_ids = torch.tensor(tags, dtype=torch.long)
            self.samples.append((word_ids, char_ids, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences within a batch"""
    words, chars, tags = zip(*batch)

    # sequence lengths
    lengths = torch.tensor([len(w) for w in words])

    # sort by length
    lengths, sort_idx = lengths.sort(descending=True)
    words = [words[i] for i in sort_idx]
    chars = [chars[i] for i in sort_idx]
    tags  = [tags[i] for i in sort_idx]

    word_padded = pad_sequence(words, batch_first=True, padding_value=0)
    tag_padded  = pad_sequence(tags,  batch_first=True, padding_value=0)

    # find max word length for char padding
    max_word_len = max(len(c) for sent in chars for c in sent)

    B, T = word_padded.shape
    char_padded = torch.zeros(B, T, max_word_len, dtype=torch.long)
    for i, seq in enumerate(chars):
        for j, word_chars in enumerate(seq):
            char_padded[i, j, :len(word_chars)] = word_chars

    return word_padded, char_padded, tag_padded, lengths