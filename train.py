import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from model.transformer.py import Transformer
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
NUM_EPOCHS = 20
LEARNING_RATE = 0.0005

# Load data
train_iterator, valid_iterator, test_iterator, SRC_VOCAB, TRG_VOCAB = load_data('data/', BATCH_SIZE)

# Initialize model, loss function, and optimizer
model = Transformer(INPUT_DIM, OUTPUT_DIM, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT)
criterion = CrossEntropyLoss(ignore_index=SRC_VOCAB.stoi['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(train_iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg[:-1, :])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[1:, :].contiguous().view(-1)

        loss = criterion(output, trg)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch: {epoch+1}, Training Loss: {epoch_loss/len(train_iterator)}')

    # Validation loop
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:-1, :])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[1:, :].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    print(f'Epoch: {epoch+1}, Validation Loss: {epoch_loss/len(valid_iterator)}')
