import os
from collections import defaultdict
from functools import partial
import torch
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>",
        }
        self.stoi = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
        }
        self.index = 4
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"

        # Backward-compatible aliases used by earlier code in the project.
        self.stoi["<SOS>"] = self.stoi[self.start_token]
        self.stoi["<EOS>"] = self.stoi[self.end_token]
        self.word2idx = self.stoi
        self.idx2word = self.itos

    def refresh_aliases(self):
        self.word2idx = self.stoi
        self.idx2word = self.itos

    def __len__(self):
        return len(self.itos)

    def _tokenize(self, sentence):
        try:
            return [token.lower() for token in word_tokenize(sentence)]
        except LookupError:
            nltk.download("punkt", quiet=True)
            return [token.lower() for token in word_tokenize(sentence)]

    def add_sentence(self, sentence):
        for token in self._tokenize(sentence):
            if token not in self.stoi:
                self.stoi[token] = self.index
                self.itos[self.index] = token
                self.index += 1

    def build_from_captions(self, captions):
        frequencies = {}

        for caption in captions:
            for token in self._tokenize(caption):
                frequencies[token] = frequencies.get(token, 0) + 1
                if frequencies[token] == self.freq_threshold and token not in self.stoi:
                    self.stoi[token] = self.index
                    self.itos[self.index] = token
                    self.index += 1

    def build_vocabulary(self, sentence_list):
        self.build_from_captions(sentence_list)

    def word_to_idx(self, word):
        return self.stoi.get(word.lower(), self.stoi[self.unk_token])

    def idx_to_word(self, index):
        return self.itos.get(index, self.unk_token)

    def numericalize(self, text):
        return [
            self.word_to_idx(word)
            for word in self._tokenize(text)
        ]


def load_captions_from_file(file_path):
    captions = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        if lines and "image" in lines[0].lower():
            lines = lines[1:]

        for line in lines:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                image_name, caption = parts
                captions.append((image_name, caption))

    return captions


class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.captions = load_captions_from_file(captions_file)
        self.transform = transform or self._default_transform()

    def _default_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        img_id, caption = self.captions[index]
        image = Image.open(os.path.join(self.image_dir, img_id)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        tokens = [self.vocab.stoi["<START>"]]
        tokens.extend(self.vocab.numericalize(caption))
        tokens.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(tokens, dtype=torch.long)


def collate_fn(data, pad_idx=0):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)
    targets = torch.nn.utils.rnn.pad_sequence(
        list(captions), batch_first=True, padding_value=pad_idx
    )

    return images, targets


def get_loader(
    image_dir,
    captions_file,
    vocab,
    batch_size=32,
    num_workers=0,
    pin_memory=True,
    transform=None,
    seed=42,
):
    dataset = FlickrDataset(image_dir, captions_file,
                            vocab, transform=transform)

    generator = torch.Generator().manual_seed(seed)

    image_to_indices = defaultdict(list)
    for index, (image_name, _) in enumerate(dataset.captions):
        image_to_indices[image_name].append(index)

    image_names = list(image_to_indices.keys())
    permutation = torch.randperm(
        len(image_names), generator=generator).tolist()
    shuffled_image_names = [image_names[i] for i in permutation]

    total_images = len(shuffled_image_names)
    train_cutoff = int(0.8 * total_images)
    val_cutoff = train_cutoff + int(0.1 * total_images)

    train_images = shuffled_image_names[:train_cutoff]
    val_images = shuffled_image_names[train_cutoff:val_cutoff]
    test_images = shuffled_image_names[val_cutoff:]

    def expand_image_names(selected_images):
        selected_indices = []
        for image_name in selected_images:
            selected_indices.extend(image_to_indices[image_name])
        return selected_indices

    train_indices = expand_image_names(train_images)
    val_indices = expand_image_names(val_images)
    test_indices = expand_image_names(test_images)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    pad_idx = vocab.stoi["<PAD>"]
    collate = partial(collate_fn, pad_idx=pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader
