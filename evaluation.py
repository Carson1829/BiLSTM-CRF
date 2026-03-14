import torch
import config
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2

def get_metrics(pred_seqs, gold_seqs, idx_to_tag):
    """Compute precision, recall, and F1"""

    pred_tags = [[idx_to_tag[i] for i in seq] for seq in pred_seqs]
    gold_tags = [[idx_to_tag[i] for i in seq] for seq in gold_seqs]

    precision = precision_score(gold_tags, pred_tags, scheme=IOB2)
    recall = recall_score(gold_tags, pred_tags, scheme=IOB2)
    f1 = f1_score(gold_tags, pred_tags, scheme=IOB2)

    return precision, recall, f1

def evaluate_model(model, val_loader, device):
    """
    Run Viterbi decoding on validation data using the DataLoader.
    Returns predicted tag sequences and gold tag sequences.
    """

    model.eval()

    preds = []
    golds = []

    with torch.no_grad():
        for word_ids, char_ids, tag_ids, lengths in val_loader:

            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            lengths = lengths.to(device)

            # Run model (Viterbi decoding)
            paths = model(word_ids, char_ids, lengths)

            # Collect predictions
            preds.extend(paths)

            # Collect gold tags (remove padding)
            for i in range(tag_ids.size(0)):
                seq_len = lengths[i].item()
                gold_seq = tag_ids[i, :seq_len].tolist()
                golds.append(gold_seq)

    return preds, golds

def write_output(tokens_list, pred_seqs, idx_to_tag, filepath):
    with open(filepath, 'w') as f:
        for tokens, preds in zip(tokens_list, pred_seqs):
            for tok, p in zip(tokens, preds):
                f.write(f"{tok}\t{idx_to_tag.get(p, 'O')}\n")
            f.write("\n")