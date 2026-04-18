from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from src.dataset import Vocabulary, get_loader, load_captions_from_file
from src.decoder import DecoderLSTM
from src.encoder import EncoderCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)


def build_vocab(captions_file):
    captions = load_captions_from_file(captions_file)
    all_sentences = [caption for _, caption in captions]
    vocab = Vocabulary(freq_threshold=CFG.vocab_freq_threshold)
    vocab.build_from_captions(all_sentences)
    return vocab


def run_validation(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions[:, :-1])
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))

            total_loss += loss.item()
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


def train():
    project_root = Path(__file__).resolve().parents[1]
    captions_file = project_root / CFG.captions_file
    images_dir = project_root / CFG.images_dir
    checkpoints_dir = project_root / CFG.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(str(captions_file))
    train_loader, val_loader, _test_loader = get_loader(
        image_dir=str(images_dir),
        captions_file=str(captions_file),
        vocab=vocab,
        batch_size=CFG.batch_size,
    )

    encoder = EncoderCNN(CFG.embed_size)
    decoder = DecoderLSTM(
        vocab_size=len(vocab),
        embed_size=CFG.embed_size,
        hidden_size=CFG.hidden_size,
        num_layers=CFG.num_layers,
        dropout=CFG.dropout,
    )
    model = ImageCaptioningModel(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.decoder.parameters(), "lr": 3e-4},
        ]
    )

    best_val_loss = float("inf")
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pth"

    for epoch in range(CFG.num_epochs):
        model.train()
        running_loss = 0.0

        for step, (images, captions) in enumerate(train_loader, start=1):
            images = images.to(device)
            captions = captions.to(device)

            # Teacher forcing: the decoder receives the ground-truth previous tokens.
            outputs = model(images, captions[:, :-1])
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

            if step % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"Epoch [{epoch + 1}/{CFG.num_epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

        val_loss = run_validation(model, val_loader, criterion, device)
        print(
            f"Epoch [{epoch + 1}/{CFG.num_epochs}] Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocabulary": vocab,
                    "vocab_stoi": vocab.stoi,
                    "vocab_itos": vocab.itos,
                    "val_loss": val_loss,
                },
                best_checkpoint_path,
            )
            print(f"Saved best checkpoint to {best_checkpoint_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
