import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import config
from model import BiLSTM_CRF
from data import load_data, build_vocab, build_tag_vocab, NERDataset, collate_fn
from evaluation import evaluate_model, get_metrics, write_output
from train import train

torch.manual_seed(1)


def main():

    # run = "OG"
    # run = "CNN"
    run = "SM"
    train_data = load_data('dataset/train.json')[:2000]
    val_data = load_data('dataset/valid.json')
    test_data  = load_data('dataset/test.json')
    print(f"Train: {len(train_data)}  Valid: {len(val_data)}  Test: {len(test_data)}")

    word_to_idx, char_to_idx = build_vocab(train_data)
    tag_to_idx, idx_to_tag   = build_tag_vocab()

    print("creating datasets")
    train_dataset = NERDataset(train_data, word_to_idx, char_to_idx, tag_to_idx)
    val_dataset = NERDataset(val_data, word_to_idx, char_to_idx, tag_to_idx)
    test_dataset = NERDataset(test_data, word_to_idx, char_to_idx, tag_to_idx)
    print("loading")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    print("creating model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_CRF(len(word_to_idx), len(char_to_idx), tag_to_idx, config.EMBEDDING_DIM, 
                       config.HIDDEN_DIM, config.CHAR_EMBED_DIM, config.CHAR_OUT_CHANNELS, run)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    print("Model created, next is training")
    f1 = train(model, train_loader, val_loader, config.EPOCHS, optimizer, device, idx_to_tag, run)
    print(f"Best F1 score after training: {f1}")

    model.load_state_dict(torch.load(f'{run}_best.pt'))
    val_tokens  = [t for t, _ in val_data]
    test_tokens = [t for t, _ in test_data]

    val_preds, val_golds = evaluate_model(model, val_loader, device)
    val_prec, val_rec, val_f1 = get_metrics(val_preds, val_golds, idx_to_tag)
    test_preds, test_golds = evaluate_model(model, test_loader, device)
    test_prec, test_rec, test_f1 = get_metrics(test_preds, test_golds, idx_to_tag)


    print(f"Dev  Precision = {val_prec:.4f}, Recall = {val_rec:.4f}, F1 = {val_f1:.4f}")
    print(f"Test Precision = {test_prec:.4f}, Recall = {test_rec:.4f}, F1 = {test_f1:.4f}")

    write_output(val_tokens,  val_preds,  idx_to_tag, f"{run}_dev.txt")
    write_output(test_tokens, test_preds, idx_to_tag, f"{run}_test.txt")
    print(f"Outputs written: {run}_dev.output, {run}_test.output")



if __name__ == "__main__":
    main()