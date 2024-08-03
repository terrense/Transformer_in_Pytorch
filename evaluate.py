import torch
from model.transformer import Transformer
from data.dataset import load_data

# Hyperparameters
INPUT_DIM = 7853
OUTPUT_DIM = 5893
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
BATCH_SIZE = 128

# Load data
train_iterator, valid_iterator, test_iterator, SRC_VOCAB, TRG_VOCAB = load_data('data/', BATCH_SIZE)

# Initialize model
model = Transformer(INPUT_DIM, OUTPUT_DIM, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT)

# Load trained model weights
model.load_state_dict(torch.load('transformer_model.pt'))

# Evaluation loop
model.eval()
epoch_loss = 0

criterion = CrossEntropyLoss(ignore_index=SRC_VOCAB.stoi['<pad>'])

with torch.no_grad():
    for i, batch in enumerate(test_iterator):
        src = batch.src
        trg = batch.trg

        output = model(src, trg[:-1, :])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[1:, :].contiguous().view(-1)

        loss = criterion(output, trg)

        epoch_loss += loss.item()

print(f'Test Loss: {epoch_loss/len(test_iterator)}')
