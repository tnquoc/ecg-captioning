
from torchvision import transforms
from torch.utils.data import DataLoader
import scipy.io as sio
from transforms import ToTensor, Resample, ApplyGain
import lorem
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from vocab import Vocabulary
from dataset import collate_fn


class RealDataset(Dataset):
    def __init__(self, length, topic, vocab, train, waveform_dir, in_length, dataset, transform, label='Label'):
        self.topic = topic
        self.dataset = dataset
        self.waveform_dir = waveform_dir
        self.in_length = in_length
        self.length = length
        self.transform = transform
        self.label = label
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')
        if train:
            self.vocab = self.setup_vocab(self.dataset['Label'])
        else:
            self.vocab = vocab

    def setup_vocab(self, labels, threshold=1):
        corpus = labels.str.cat(sep=" ")

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        counter = counter.most_common()
        words = []

        for i in range(0, len(counter)):
            words.append(counter[i][0])

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')
        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab.add_word(word)

        return vocab

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        waveform, sample_id = self.get_waveform(idx)

        sample = {
            'waveform': waveform,
            'id': sample_id,
        }

        if self.label in self.dataset.columns.values:
            sentence = self.dataset[self.label].iloc[idx]
            tokens = self.tokenizer.tokenize(sentence)
            vocab = self.vocab
            caption = [vocab('<start>')]
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)

            sample['label'] = target

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_waveform(self, idx):
        waveform = sio.loadmat(self.waveform_dir + '/' + self.dataset['EventID'][idx])['ECG'][0][0][2][1]
        while True:
            if waveform.shape[0] > self.in_length:
                waveform = waveform[:self.in_length]
            elif waveform.shape[0] < self.in_length:
                waveform = np.append(waveform, waveform[:self.in_length - waveform.shape[0]], axis=0)
            else:
                break
        waveform = np.array(waveform.T)
        waveform = np.nan_to_num(waveform)

        return waveform, idx


class FakeDataset:
    def __init__(self, length, topic, vocab, transform):
        self.length = length
        self.topic = topic
        self.transform = transform
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')

        if vocab is None:
            self.vocab = self.setup_vocab(0)
        else:
            self.vocab = vocab

    def setup_vocab(self, threshold):
        corpus = " ".join([lorem.sentence() for _ in range(self.length)])

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        words = [word for word, cnt in counter.items() if cnt >= threshold]

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        # Add the words to the vocabulary.
        for _, word in enumerate(words):
            vocab.add_word(word)
        return vocab

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        waveform = np.random.rand(12, 5000)

        sample = {'waveform': waveform,
                  'samplebase': 500,
                  'gain': 4.88,
                  'id': int(np.random.rand(1)),
                  }

        sentence = lorem.sentence()

        try:
            tokens = self.tokenizer.tokenize(sentence)
        except:
            print(sentence)
            raise Exception()

        vocab = self.vocab
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        sample['label'] = target

        if self.topic:
            topic_label_classes = 100

            topic_labels_bools = np.random.randint(2, size=topic_label_classes)
            topic_tensor = torch.from_numpy(topic_labels_bools).float()
            topic_tensor_norm = topic_tensor / topic_tensor.sum()

            sample['extra_label'] = topic_tensor_norm

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_loaders(params, topic):
    transform = transforms.Compose([ToTensor()])

    train_df = pd.read_csv(params['train_labels_csv'])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(params['val_labels_csv'])
    val_df = val_df.sample(frac=1).reset_index(drop=True)

    is_train, vocab = True, None
    trainset = RealDataset(len(train_df), topic, vocab, is_train, params['data_dir'], params['in_length'], train_df,
                           transform=transform)

    is_train, vocab = False, trainset.vocab
    valset = RealDataset(len(val_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                         val_df, transform=transform)

    testset_df = pd.read_csv(params['test_labels_csv'])
    testset = RealDataset(len(testset_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                          testset_df, transform=transform)

    train_loader = DataLoader(trainset, batch_size=params['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)

    val_loader = DataLoader(valset, batch_size=params['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(testset, batch_size=params['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab


def get_loaders_toy_data(params, topic=False):
    vocab = None
    transform = transforms.Compose([Resample(500), ToTensor(), ApplyGain()])
    train_set = FakeDataset(1000, topic, vocab, transform)
    vocab = train_set.vocab
    val_set = FakeDataset(200, topic, vocab, transform)
    test_set = FakeDataset(200, topic, vocab, transform)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=params['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(test_set, batch_size=params['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab
