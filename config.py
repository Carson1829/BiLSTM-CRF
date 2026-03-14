START_TAG = '<START>'
STOP_TAG  = '<STOP>'
PAD_WORD  = '<PAD>'
UNK_WORD  = '<UNK>'
PAD_CHAR  = '<PAD_CHAR>'
UNK_CHAR  = '<UNK_CHAR>'

LABEL_NAMES = ['O', 'B-DNA', 'I-DNA', 'B-protein', 'I-protein', 'B-cell_line',
    'I-cell_line', 'B-cell_type', 'I-cell_type', 'B-RNA', 'I-RNA']

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 25
DROPOUT = 0.5
COST_FACTOR = 1.0
CHAR_OUT_CHANNELS = 50
CHAR_EMBED_DIM = 30