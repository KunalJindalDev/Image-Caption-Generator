from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from config import CFG
from src.dataset import Vocabulary, get_loader
from src.decoder_attention import DecoderWithAttention
from src.encoder_attention import EncoderAttentionCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECIAL_TOKENS = {"<PAD>", "<START>", "<END>", "<UNK>"}


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in {"<pad>", "<start>", "<end>", "<unk>"}:
            continue
        normalized.append(lowered)
    return normalized


def decode_generated_tokens(token_ids: Iterable[int], vocab: Vocabulary) -> List[str]:
    tokens: List[str] = []
    for token_id in token_ids:
        word = vocab.idx_to_word(int(token_id))
        if word == "<END>":
            break
        if word in SPECIAL_TOKENS:
            continue
        tokens.append(word)
    return normalize_tokens(tokens)


def tensor_caption_to_tokens(caption_tensor: torch.Tensor, vocab: Vocabulary) -> List[str]:
    tokens: List[str] = []
    for idx in caption_tensor.tolist():
        word = vocab.idx_to_word(int(idx))
        if word == "<END>":
            break
        if word in SPECIAL_TOKENS:
            continue
        tokens.append(word)
    return normalize_tokens(tokens)


def load_attention_checkpoint(checkpoint_path: Path) -> tuple[EncoderAttentionCNN, DecoderWithAttention, Vocabulary]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "vocabulary" not in checkpoint:
        raise ValueError(
            "Checkpoint missing 'vocabulary'. Expected attention checkpoint from train_attention.py"
        )

    vocab = checkpoint["vocabulary"]
    if hasattr(vocab, "refresh_aliases"):
        vocab.refresh_aliases()

    encoder = EncoderAttentionCNN(encoder_dim=CFG.hidden_size).to(device)
    decoder = DecoderWithAttention(
        vocab_size=len(vocab),
        embed_dim=CFG.embed_size,
        encoder_dim=CFG.hidden_size,
        decoder_dim=CFG.hidden_size,
        dropout=CFG.dropout,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab


def gather_unique_test_samples(test_loader, vocab: Vocabulary):
    subset = test_loader.dataset
    dataset = subset.dataset
    indices = subset.indices

    grouped: Dict[str, Tuple[torch.Tensor, List[List[str]]]] = {}
    for original_index in indices:
        image_name, _ = dataset.captions[original_index]
        image_tensor, caption_tensor = dataset[original_index]
        reference_tokens = tensor_caption_to_tokens(caption_tensor, vocab)

        if image_name not in grouped:
            grouped[image_name] = (image_tensor.cpu(), [reference_tokens])
        else:
            image_ref, refs = grouped[image_name]
            refs.append(reference_tokens)
            grouped[image_name] = (image_ref, refs)

    image_names = list(grouped.keys())
    image_tensors = [grouped[name][0] for name in image_names]
    references = [grouped[name][1] for name in image_names]
    return image_tensors, references


@torch.no_grad()
def generate_hypotheses(
    encoder: EncoderAttentionCNN,
    decoder: DecoderWithAttention,
    image_tensors: List[torch.Tensor],
    vocab: Vocabulary,
    beam_size: int,
) -> List[List[str]]:
    hypotheses: List[List[str]] = []

    batch_size = CFG.batch_size
    for start in range(0, len(image_tensors), batch_size):
        batch_images = torch.stack(
            image_tensors[start:start + batch_size], dim=0).to(device)
        encoder_out = encoder(batch_images)

        if beam_size == 1:
            token_ids, _alphas = decoder.generate(
                encoder_out,
                max_len=CFG.max_caption_length,
                start_idx=vocab.stoi["<START>"],
                end_idx=vocab.stoi["<END>"],
            )
        else:
            token_ids, _alphas = decoder.generate_beam(
                encoder_out,
                beam_size=beam_size,
                max_len=CFG.max_caption_length,
                start_idx=vocab.stoi["<START>"],
                end_idx=vocab.stoi["<END>"],
            )

        for row in token_ids.cpu():
            hypotheses.append(decode_generated_tokens(row.tolist(), vocab))

    return hypotheses


def compute_bleu_scores(references: List[List[List[str]]], hypotheses: List[List[str]]) -> tuple[float, float, float, float]:
    smoothing = SmoothingFunction().method1

    bleu_1 = float(
        corpus_bleu(references, hypotheses, weights=(
            1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing)
    )
    bleu_2 = float(
        corpus_bleu(references, hypotheses, weights=(
            0.5, 0.5, 0.0, 0.0), smoothing_function=smoothing)
    )
    bleu_3 = float(
        corpus_bleu(
            references,
            hypotheses,
            weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
            smoothing_function=smoothing,
        )
    )
    bleu_4 = float(
        corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
    )
    return bleu_1, bleu_2, bleu_3, bleu_4


def print_baseline_vs_attention(attention_scores: tuple[float, float, float, float]):
    baseline = (0.5897, 0.3821, 0.2501, 0.1623)
    print("\nExperiment 1 - Baseline vs Attention")
    print("+------------------------+--------+--------+--------+--------+")
    print("| Model                  | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |")
    print("+------------------------+--------+--------+--------+--------+")
    print(
        f"| CNN+LSTM (baseline)    | {baseline[0]:.4f} | {baseline[1]:.4f} | {baseline[2]:.4f} | {baseline[3]:.4f} |"
    )
    print(
        f"| CNN+LSTM+Attention     | {attention_scores[0]:.4f} | {attention_scores[1]:.4f} | {attention_scores[2]:.4f} | {attention_scores[3]:.4f} |"
    )
    print("+------------------------+--------+--------+--------+--------+")


def print_beam_table(rows: List[tuple[int, tuple[float, float, float, float]]]):
    print("\nExperiment 2 - Greedy vs Beam Search")
    print("+-----------+--------+--------+--------+--------+")
    print("| Beam Size | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |")
    print("+-----------+--------+--------+--------+--------+")
    for beam_size, scores in rows:
        print(
            f"| {beam_size:^9} | {scores[0]:.4f} | {scores[1]:.4f} | {scores[2]:.4f} | {scores[3]:.4f} |"
        )
    print("+-----------+--------+--------+--------+--------+")


def evaluate_attention(checkpoint_path: Path, beam_sizes: List[int]):
    project_root = Path(__file__).resolve().parents[1]
    captions_file = project_root / CFG.captions_file
    images_dir = project_root / CFG.images_dir

    encoder, decoder, vocab = load_attention_checkpoint(checkpoint_path)

    _, _, test_loader = get_loader(
        image_dir=str(images_dir),
        captions_file=str(captions_file),
        vocab=vocab,
        batch_size=CFG.batch_size,
    )

    image_tensors, references = gather_unique_test_samples(test_loader, vocab)
    beam_rows: List[tuple[int, tuple[float, float, float, float]]] = []

    for beam_size in beam_sizes:
        hypotheses = generate_hypotheses(
            encoder=encoder,
            decoder=decoder,
            image_tensors=image_tensors,
            vocab=vocab,
            beam_size=beam_size,
        )
        scores = compute_bleu_scores(references, hypotheses)
        beam_rows.append((beam_size, scores))

    beam_rows.sort(key=lambda x: x[0])
    print_beam_table(beam_rows)

    greedy_row = next((row for row in beam_rows if row[0] == 1), None)
    if greedy_row is not None:
        print_baseline_vs_attention(greedy_row[1])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate attention model BLEU with greedy and beam search.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve(
        ).parents[1] / CFG.checkpoints_dir / "best_attention_checkpoint.pth"),
        help="Path to attention checkpoint.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Beam sizes to evaluate. Use 1 for greedy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    beam_sizes = sorted(set(args.beam_size))
    evaluate_attention(Path(args.checkpoint), beam_sizes)


if __name__ == "__main__":
    main()
