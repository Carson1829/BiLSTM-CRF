START_TAG = '<START>'
STOP_TAG  = '<STOP>'
PAD_WORD  = '<PAD>'
UNK_WORD  = '<UNK>'
PAD_CHAR  = '<PAD_CHAR>'
UNK_CHAR  = '<UNK_CHAR>'

LABEL_NAMES = ['O', 'B-DNA', 'I-DNA', 'B-protein', 'I-protein', 'B-cell_line',
    'I-cell_line', 'B-cell_type', 'I-cell_type', 'B-RNA', 'I-RNA']

EMBEDDING_DIM = 5
HIDDEN_DIM = 4
LR = 0.003
BATCH_SIZE = 32
EPOCHS = 5
DROPOUT = 0.1
COST_FACTOR = 1.0
CHAR_OUT_CHANNELS = 50
CHAR_EMBED_DIM = 30