from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from src.dataset import Vocabulary, get_loader, load_captions_from_file
from src.decoder_attention import DecoderWithAttention
from src.encoder_attention import EncoderAttentionCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vocab(captions_file):
    captions = load_captions_from_file(captions_file)
    all_sentences = [caption for _, caption in captions]
    freq_threshold = getattr(CFG, "vocab_freq_threshold", 5)
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_from_captions(all_sentences)
    return vocab


def build_vocab_from_train_split(captions_file):
    captions = load_captions_from_file(captions_file)
    image_to_captions = defaultdict(list)
    for image_name, caption in captions:
        image_to_captions[image_name].append(caption)

    image_names = list(image_to_captions.keys())
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(
        len(image_names), generator=generator).tolist()
    shuffled_image_names = [image_names[i] for i in permutation]

    train_cutoff = int(0.8 * len(shuffled_image_names))
    train_images = shuffled_image_names[:train_cutoff]

    train_captions = [
        caption
        for image_name in train_images
        for caption in image_to_captions[image_name]
    ]

    freq_threshold = getattr(CFG, "vocab_freq_threshold", 5)
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_from_captions(train_captions)
    return vocab


def run_validation(encoder, decoder, data_loader, criterion, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            captions = captions.to(device)

            encoder_out = encoder(images)
            decoder_logits, _alphas = decoder(encoder_out, captions[:, :-1])
            loss = criterion(
                decoder_logits.reshape(-1, decoder_logits.size(-1)),
                captions[:, 1:].reshape(-1),
            )

            total_loss += loss.item()
            total_steps += 1

    encoder.train()
    decoder.train()
    return total_loss / max(total_steps, 1)


def train():
    project_root = Path(__file__).resolve().parents[1]
    captions_file = project_root / CFG.captions_file
    images_dir = project_root / CFG.images_dir
    checkpoints_dir = project_root / CFG.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    num_workers = getattr(CFG, "num_workers",
                          2 if device.type == "cuda" else 0)
    log_every = getattr(CFG, "log_every", 50)
    use_amp = bool(getattr(CFG, "use_amp", device.type == "cuda"))
    amp_enabled = use_amp and device.type == "cuda"

    vocab = build_vocab_from_train_split(str(captions_file))
    train_loader, val_loader, _test_loader = get_loader(
        image_dir=str(images_dir),
        captions_file=str(captions_file),
        vocab=vocab,
        batch_size=CFG.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    encoder = EncoderAttentionCNN(encoder_dim=CFG.hidden_size).to(device)
    decoder = DecoderWithAttention(
        vocab_size=len(vocab),
        embed_dim=CFG.embed_size,
        encoder_dim=CFG.hidden_size,
        decoder_dim=CFG.hidden_size,
        dropout=CFG.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)
    optimizer = optim.Adam(
        [
            {"params": encoder.parameters(), "lr": 1e-4},
            {"params": decoder.parameters(), "lr": 4e-4},
        ]
    )

    best_val_loss = float("inf")
    best_checkpoint_path = checkpoints_dir / "best_attention_checkpoint.pth"
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    num_epochs = 20

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        for step, (images, captions) in enumerate(train_loader, start=1):
            images = images.to(device)
            captions = captions.to(device)

            # Teacher forcing: feed ground-truth tokens up to T-1.
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                encoder_out = encoder(images)
                decoder_logits, _alphas = decoder(
                    encoder_out, captions[:, :-1])
                loss = criterion(
                    decoder_logits.reshape(-1, decoder_logits.size(-1)),
                    captions[:, 1:].reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if step == 1 or step % log_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Train Loss: {loss.item():.4f}"
                )

        val_loss = run_validation(
            encoder, decoder, val_loader, criterion, device)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
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
