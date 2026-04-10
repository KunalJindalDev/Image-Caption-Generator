import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class Vocabulary:
    def __init__(self, freq_threshold=5):
        # We only keep words that appear at least 'freq_threshold' times.
        # This prevents typos or super rare words from confusing the AI.
        self.freq_threshold = freq_threshold
        
        # stoi: String TO Integer (word -> number)
        # itos: Integer TO String (number -> word)
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        
        # Start counting at 4 because 0-3 are taken by our special tokens!
        self.index = 4 

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        # This reads through all your captions and builds the dictionary
        frequencies = {}
        
        for sentence in sentence_list:
            for word in sentence.lower().split():
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                # If the word shows up enough times, add it to our official dictionary
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = self.index
                    self.itos[self.index] = word
                    self.index += 1

    def numericalize(self, text):
        # This takes a fresh sentence and translates it into a list of numbers
        tokenized_text = text.lower().split()
        
        return [
            self.stoi[word] if word in self.stoi else self.stoi["<UNK>"]
            for word in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_dict, vocab, transform=None):
        self.image_dir = image_dir
        # captions_dict is a list of tuples: (image_filename, caption_text)
        self.captions = captions_dict 
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        img_id, caption = self.captions[index]
        
        # Load and transform the image
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Convert the text sentence into a list of numbers using your Vocabulary
        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += [self.vocab.stoi[word] for word in caption.lower().split()]
        tokens += [self.vocab.stoi["<EOS>"]]
        
        return image, torch.tensor(tokens)

# This function handles the "padding" for different sentence lengths
def collate_fn(data):
    # Sort the batch by caption length (longest first)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    
    # zip(*data) returns tuples!
    images, captions = zip(*data)

    # Stack images into a single 4D tensor
    images = torch.stack(images, 0)
    
    # FIX: Convert the tuple of text tensors into a standard Python list
    captions_list = list(captions)
    
    # Now pad the sequence using the list
    targets = torch.nn.utils.rnn.pad_sequence(captions_list, batch_first=True, padding_value=0)
    
    return images, targets