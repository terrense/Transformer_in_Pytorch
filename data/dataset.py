import torch
from torchtext.data import Field, BucketIterator, TabularDataset

# Define the fields for source and target sequences
SRC = Field(tokenize = str.split, init_token = '<sos>', eos_token = '<eos>', lower = True)
TRG = Field(tokenize = str.split, init_token = '<sos>', eos_token = '<eos>', lower = True)

def load_data(path, batch_size):
    """
    Load data from a specified path and create data iterators.

    Args:
        path (str): The path to the data file.
        batch_size (int): The batch size for the data iterators.

    Returns:
        Tuple of train, validation, and test iterators along with the source and target vocabularies.
    """
    data_fields = [('src', SRC), ('trg', TRG)]

    train_data, valid_data, test_data = TabularDataset.splits(
        path = path,
        train = 'train.csv',
        validation = 'valid.csv',
        test = 'test.csv',
        format = 'csv',
        fields = data_fields
    )

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    return train_iterator, valid_iterator, test_iterator, SRC.vocab, TRG.vocab
