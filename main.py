from curses import version
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils.rnn as rnn_utils
from Models.ML_VALLR import ML_VALLR
from Models.VALLR import VALLR
from Models.ctc_decode import (
    CTCHypothesis,
    greedy_decode,
    topk_beam_decode,
    format_prompt_for_llm,
)
from Data.dataset import VideoDataset
import torch.nn as nn
from config import load_args, WarmupScheduler, get_vocab
import numpy as np
from transformers import VideoMAEConfig, Wav2Vec2Config, Wav2Vec2Tokenizer, GPT2Tokenizer, GPT2LMHeadModel
import os
import json
import numpy as np
from decord import VideoReader, cpu
import torch.nn.functional as F
import jiwer
from torch.amp import autocast, GradScaler
from itertools import zip_longest

args = load_args()

def monitor_gradients(model):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"{name} grad norm: {param_norm.item()}")
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")
    return total_norm

# Function to clamp logits
def clamp_logits(logits, min_logit=-10, max_logit=10):
    """
    Clamps the logits to prevent numerical instability.
    Args:
        logits (Tensor): The logits output from the model.
        min_logit (float): Minimum logit value.
        max_logit (float): Maximum logit value.
    Returns:
        Tensor: Clamped logits.
    """
    return torch.clamp(logits, min_logit, max_logit)

def custom_collate(batch, phoneme_vocab):
    """
    Custom collate function to handle batches of videos and labels.
    Filters out None values from the batch, and returns the valid entries.
    """
    # Filter out any None entries from the batch
    valid_batch = [item for item in batch if item is not None]

    # If no valid items are left after filtering, return a tensor with an empty batch
    if len(valid_batch) == 0:
        # Return an empty tensor instead of None
        return torch.empty(0), torch.empty(0)

    # Unpack valid_batch into videos and labels
    videos, labels = zip(*valid_batch)

    # Convert videos into a single tensor
    videos = torch.stack(videos)

    # Convert labels to a list of tensors
    label_tensors = [label.clone().detach() for label in labels]

    # Return videos and the list of label tensors without padding
    return videos, label_tensors

def log_all_metrics(epoch, epochs, train_loss, train_acc, val_loss, val_acc, lr):
    """
    Log all relevant metrics at once for a single epoch.
    Args:
        epoch (int): The current epoch number.
        epochs (int): The total number of epochs.
        train_loss (float): The training loss for the current epoch.
        train_acc (float): The training accuracy for the current epoch.
        val_loss (float): The validation loss for the current epoch.
        val_acc (float): The validation accuracy for the current epoch.
        lr (float): The current learning rate.
    """
    # Log all metrics for the epoch
    print(f"Epoch [{epoch}/{epochs}], "
          f"Training Loss: {train_loss:.4f}, "
          f"Training Accuracy: {train_acc:.2f}%, "
          f"Validation Loss: {val_loss:.4f}, "
          f"Validation Accuracy: {val_acc:.2f}%, "
          f"Learning Rate: {lr:.8f}")

    # Optionally log metrics to WandB or other tracking tools
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": lr
    })

def train_one_epoch(model, dataloader, optimizer, criterion, device, phoneme_vocab):
    """Train the model for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    scaler = GradScaler()  # Mixed precision gradient scaler

    reverse_vocab = {v: k for k, v in phoneme_vocab.items()}  # Reverse mapping from index to phoneme for decoding

    for batch in tqdm(dataloader, desc="Training", leave=False):
        videos, labels = batch

        if videos.size(0) == 0:  # Check if the batch is empty
            print("Skipping empty batch")
            continue

        labels = [label.to(device) for label in labels]
        videos = videos.to(device)
        videos = videos.float()

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast('cuda'):
            raw_logits, _ = model(videos)  # Raw logits, ignore featuress)
            raw_logits = raw_logits.log_softmax(dim=-1)  # Apply log_softmax to get log probabilities
            transpose_logits = raw_logits.transpose(0, 1)  # (time_steps, batch_size, num_classes)

            batch_size = transpose_logits.size(1)
            input_lengths = torch.full(size=(batch_size,), fill_value=transpose_logits.size(0), dtype=torch.long).to(device)
            target_lengths = torch.tensor([label.size(0) for label in labels], dtype=torch.long).to(device)

            if input_lengths.min() < target_lengths.max():
                print(f"Skipping batch: input lengths {input_lengths.min()} < target lengths {target_lengths.max()}")
                max_len = target_lengths.max().item()
                max_idx = torch.argmax(target_lengths).item()
                # print(f"⚠️ Skipping batch: input lengths {input_lengths.min().item()} < target length {max_len}")
                print(f"Longest target length index: {max_idx}")
                print(f"Target tensor (longest): {labels[max_idx].tolist()}")
                # quit()
                continue  # Skip the batch if the input lengths are too short

            loss = criterion(transpose_logits, torch.cat(labels), input_lengths, target_lengths)

            if loss.item() < 0.0:
                print("Negative loss detected. Skipping...")
                continue

            # Backward pass with mixed precision scaling
            scaler.scale(loss).backward()

            # Gradient clipping 
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Perform gradient clipping

            # Step with scaled gradients
            scaler.step(optimizer)

            # Update the scale for the next iteration
            scaler.update()

            running_loss += loss.item()

        # Decode predicted outputs (using argmax)
        predicted_indices = torch.argmax(raw_logits, dim=-1)

        predicted_phonemes = []
        for batch_idx in range(predicted_indices.size(0)):  # Iterate over batch dimension
            frame_sequence = predicted_indices[batch_idx].tolist()  # Get predictions for the current sequence

            decoded_phoneme_seq = []
            previous_phoneme = None

            for timestep_idx in frame_sequence:
                phoneme = reverse_vocab.get(timestep_idx, None)

                # Skip blank tokens and repeated phonemes
                if phoneme == '<pad>' or phoneme == previous_phoneme:
                    continue

                decoded_phoneme_seq.append(phoneme)
                previous_phoneme = phoneme

            predicted_phonemes.append(decoded_phoneme_seq)

        # print("Predicted Phonemes", predicted_phonemes)

        true_phonemes = []
        for label_seq in labels:
            phoneme_seq = [reverse_vocab[idx.item()] for idx in label_seq if idx.item() in reverse_vocab]
            true_phonemes.append(phoneme_seq)

        # Calculate accuracy by comparing the predicted and true phonemes
        for pred_seq, true_seq in zip_longest(predicted_phonemes, true_phonemes, fillvalue=[]):
            # Ensure that zip_longest doesn't introduce issues with None values
            correct_predictions = sum(p == t for p, t in zip_longest(pred_seq, true_seq, fillvalue=None) if p is not None and t is not None)
            
            # Add only non-None length to total_samples
            total_correct += correct_predictions
            total_samples += len([t for t in true_seq if t is not None])

    avg_loss = running_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    torch.cuda.empty_cache()  # Clear CUDA memory

    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, criterion, device, phoneme_vocab):
    """Validate the model for one epoch with mixed precision."""
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    reverse_vocab = {v: k for k, v in phoneme_vocab.items()}  # Reverse mapping from index to phoneme for decoding

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            videos, labels = batch

            if videos.size(0) == 0:  # Check if the batch is empty
                print("Skipping empty batch")
                continue

            labels = [label.to(device) for label in labels]
            videos = videos.to(device)
            videos = videos.float()

            with autocast('cuda'):  # Use mixed precision here as well
                raw_logits, _ = model(videos)  # Raw logits, ignore features
                raw_logits = raw_logits.log_softmax(dim=-1)  # Apply log_softmax to get log probabilities
                transpose_logits = raw_logits.transpose(0, 1)  # (time_steps, batch_size, num_classes)

                batch_size = transpose_logits.size(1)
                input_lengths = torch.full(size=(batch_size,), fill_value=transpose_logits.size(0), dtype=torch.long).to(device)
                target_lengths = torch.tensor([label.size(0) for label in labels], dtype=torch.long).to(device)

                if input_lengths.min() < target_lengths.max():
                    print(f"Skipping batch: input lengths {input_lengths.min()} < target lengths {target_lengths.max()}")
                    continue  # Skip the batch if the input lengths are too short

                loss = criterion(transpose_logits, torch.cat(labels), input_lengths, target_lengths)

                if loss.item() < 0.0:
                    print("Negative loss detected. Skipping...")
                    continue

            running_loss += loss.item()

            # Decode predicted outputs (using argmax)
            predicted_indices = torch.argmax(raw_logits, dim=-1)

            predicted_phonemes = []
            for batch_idx in range(predicted_indices.size(0)):  # Iterate over batch dimension
                frame_sequence = predicted_indices[batch_idx].tolist()  # Get predictions for the current sequence

                decoded_phoneme_seq = []
                previous_phoneme = None

                for timestep_idx in frame_sequence:
                    phoneme = reverse_vocab.get(timestep_idx, None)

                    # Skip blank tokens and repeated phonemes
                    if phoneme == '<pad>' or phoneme == previous_phoneme:
                        continue

                    decoded_phoneme_seq.append(phoneme)
                    previous_phoneme = phoneme

                predicted_phonemes.append(decoded_phoneme_seq)

            # print("Predicted Phonemes", predicted_phonemes)

            true_phonemes = []
            for label_seq in labels:
                phoneme_seq = [reverse_vocab[idx.item()] for idx in label_seq if idx.item() in reverse_vocab]
                true_phonemes.append(phoneme_seq)

            for pred_seq, true_seq in zip_longest(predicted_phonemes, true_phonemes, fillvalue=[]):
                # Ensure that zip_longest doesn't introduce issues with None values
                correct_predictions = sum(p == t for p, t in zip_longest(pred_seq, true_seq, fillvalue=None) if p is not None and t is not None)
                
                # Add only non-None length to total_samples
                total_correct += correct_predictions
                total_samples += len([t for t in true_seq if t is not None])

    avg_loss = running_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    torch.cuda.empty_cache()  # Clear CUDA memory

    return avg_loss, accuracy

def save_model(model, save_model_path):
    """Save the model to the specified path."""
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

def _load_encoder_cfg(encoder_type: str, multiscale_config_path: str):
    """Load an encoder config dict from an optional JSON file."""
    if encoder_type != "multiscale":
        return None
    if multiscale_config_path and os.path.exists(multiscale_config_path):
        with open(multiscale_config_path, "r") as f:
            return json.load(f)
    # Fall back to module defaults; signal multiscale selection.
    return {"encoder_type": "multiscale"}


def train(
    device,
    version,
    video_path,
    batch_size,
    num_workers,
    epochs,
    save_model_path,
    sample_size,
    vocab,
    encoder_type: str = "adapter",
    multiscale_config_path: str = None,
):
    """Main training function."""

    torch.cuda.empty_cache()  # Clear CUDA memory

    # Extract the pretrained vocabulary
    phoneme_vocab = vocab # tokenizer.get_vocab()
    encoder_cfg = _load_encoder_cfg(encoder_type, multiscale_config_path)

    # Initialize model
    if version == "V1":
        if encoder_type != "adapter":
            print(
                f"[warn] encoder_type={encoder_type!r} is only supported on version V2; "
                "falling back to the original V1 adapter path."
            )
        videomae_config = VideoMAEConfig()  # Use default configuration
        wav2vec_config = Wav2Vec2Config()
        wav2vec_config.vocab_size = len(vocab)

        model= VALLR(
            videomae_config=videomae_config,
            wav2vec_config=wav2vec_config,
            adapter_dim=256,
        )
    elif version == "V2":
        model = ML_VALLR(
            adapter_dim=256,                    # Adapter dimension # Based on the sampling frequency of the video
            num_classes=len(phoneme_vocab),
            encoder_type=encoder_type,
            encoder_config=encoder_cfg,
        )

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion = nn.CTCLoss(blank=phoneme_vocab['<pad>'], reduction="mean", zero_infinity=True)
    lr = 1e-6  # initial learning rate
    target_lr = 1e-4  # final learning rate after warm-up
    warmup_steps = 500  # define your warmup steps

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    after_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Set up warm-up scheduler
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        base_lr=lr,
        target_lr=target_lr,
        after_scheduler=after_scheduler
    )

    # Load datasets and create DataLoader
    training_dataset = VideoDataset(
        video_dir=video_path,
        split="train",
        num_frames=16,
        frame_size=(224, 224),
        phoneme_vocab=phoneme_vocab  # Include phoneme_vocab argument
    )

    validation_dataset = VideoDataset(
        video_dir=video_path,
        split="val",
        num_frames=16,
        frame_size=(224, 224),
        phoneme_vocab=phoneme_vocab  # Include phoneme_vocab argument
    )

    # Load datasets and create DataLoader
    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab)
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab)
    )

    # Create a subset of the training and validation datasets as a percentage
    subset_percentage = sample_size  # Define the subset percentage (e.g., 10%)
    train_subset_size = int(len(training_dataset) * subset_percentage)
    val_subset_size = int(len(validation_dataset) * subset_percentage)

    train_subset_indices = torch.randperm(len(training_dataset))[:train_subset_size]
    val_subset_indices = torch.randperm(len(validation_dataset))[:val_subset_size]

    train_subset = torch.utils.data.Subset(training_dataset, train_subset_indices)
    val_subset = torch.utils.data.Subset(validation_dataset, val_subset_indices)

    # Create DataLoader for the subsets
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab)
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab)
    )

    # Start a new wandb run
    wandb.init(
        project="VALLR",
        config={
            "learning_rate": lr,
            "architecture": "VALLR",
            "dataset": "Custom Dataset",
            "epochs": epochs,
        }
    )

    model.to(device)

    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        epoch_train_loss, epoch_train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, phoneme_vocab)

        # Validate for one epoch
        epoch_val_loss, epoch_val_acc = validate_one_epoch(model, val_loader, criterion, device, phoneme_vocab)

        # Log all metrics for the epoch
        current_lr = scheduler.get_last_lr()[0]
        # current_lr = after_scheduler.get_last_lr()[0]
        log_all_metrics(epoch+1, epochs, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc, current_lr)
        # Step the scheduler with the validation loss
        scheduler.step(epoch_val_loss)
        # after_scheduler.step(epoch_val_loss)
        save_model(model, save_model_path)

    # Finish Wandb run and save the model
    wandb.finish()

def load_videos(video_path, num_frames=16, frame_size=(224, 224)):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    except Exception as e:
        print(f"Error loading video: {video_path}. Error: {e}")
        return None

    frame_count = len(vr)
    if frame_count < num_frames:
        print(f"Warning: Not enough frames in video {video_path}. Skipping.")
        return None

    # Sample indices for the frames evenly spaced across the video
    sample_indices = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in sample_indices:
        frame = vr[idx].asnumpy()

        # Convert frame from HWC (height, width, channel) to CHW (channel, height, width) format
        frame = np.transpose(frame, (2, 0, 1))

        frames.append(frame)

    if len(frames) < num_frames:
        return None

    # Convert frames to a NumPy array and then to a tensor
    video_np = np.array(frames)  # Shape: (T, C, H, W)
    video_tensor = torch.tensor(video_np).float()  # Convert to float tensor

    # Add the batch dimension (for a single video) and return it
    video_tensor = video_tensor.unsqueeze(0)  # Shape: (1, T, C, H, W)

    return video_tensor

def load_finetuned_model(
    model_path,
    device,
    version,
    vocab,
    encoder_type: str = "adapter",
    multiscale_config_path: str = None,
):
    """
    Load the fine-tuned model from the given checkpoint.
    Args:
        model_path: Path to the saved model.
        device: Device to load the model (CPU or GPU).
        phoneme_vocab: The phoneme vocabulary for the model.
        encoder_type: ``"adapter"`` (baseline) or ``"multiscale"``.
        multiscale_config_path: Optional JSON config path.
    Returns:
        model: The fine-tuned VT4SM model.
    """
    # Extract the pretrained vocabulary
    phoneme_vocab = vocab # tokenizer.get_vocab()
    encoder_cfg = _load_encoder_cfg(encoder_type, multiscale_config_path)

    # Initialize model
    if version == "V1":
        videomae_config = VideoMAEConfig()  # Use default configuration
        # wav2vec_config = Wav2Vec2Config()  # Use default configuration
        wav2vec_config = Wav2Vec2Config()
        wav2vec_config.vocab_size = len(vocab)

        model= VALLR(
            videomae_config=videomae_config,
            wav2vec_config=wav2vec_config,
            adapter_dim=256,
        )
    elif version == "V2":
        model = ML_VALLR(
            adapter_dim=256,  # Adjust this based on your adapter dimensions
            num_classes=len(phoneme_vocab),
            encoder_type=encoder_type,
            encoder_config=encoder_cfg,
        )

    # Load the saved state dict (fine-tuned model weights)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move the model to the device (CPU or GPU)
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")

    return model

def _load_json_cfg(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def run_inference(
    model_path,
    version,
    video_path,
    device,
    phoneme_vocab,
    beam_width=3,
    top_k: int = 1,
    ctc_beam_width: int = 16,
    topk_config_path: str = None,
    encoder_type: str = "adapter",
    multiscale_config_path: str = None,
):
    """
    Run inference using the fine-tuned model.

    Backward-compatibility contract:
      * ``top_k == 1`` reproduces the original one-best path and the
        returned structure is unchanged: ``(list[list[str]], feats)``.
      * ``top_k > 1`` returns ``(list[list[CTCHypothesis]], feats,
        list[str])`` where the third element is a per-example LLM
        prompt already formatted with the top-k hypotheses.

    Args:
        model_path: Path to the saved model.
        video_path: Path to the video file.
        device: Device to run the inference on (CPU or GPU).
        beam_width: Legacy argument kept for backwards compatibility.
        top_k: Number of CTC hypotheses. ``1`` = baseline one-best.
        ctc_beam_width: Prefix-beam width when ``top_k > 1``.
        topk_config_path: Optional JSON with top-k/prompt overrides.
    """
    # Load the video inputs with the correct number of frames
    video_inputs = load_videos(video_path, num_frames=16)

    if video_inputs is None:
        print("Error: Failed to load video frames.")
        return []

    # Load the fine-tuned model
    model = load_finetuned_model(
        model_path,
        device,
        version,
        phoneme_vocab,
        encoder_type=encoder_type,
        multiscale_config_path=multiscale_config_path,
    )

    # Load the phoneme vocabulary
    reverse_vocab = {v: k for k, v in phoneme_vocab.items()}  # Reverse mapping from index to phoneme
    blank_id = phoneme_vocab.get('<pad>', 0)

    # Optional overrides from JSON config.
    topk_cfg = _load_json_cfg(topk_config_path)
    if topk_cfg:
        top_k = int(topk_cfg.get("top_k", top_k))
        ctc_beam_width = int(topk_cfg.get("beam_width", ctc_beam_width))
    prune_log_prob = float(topk_cfg.get("prune_log_prob", -6.0)) if topk_cfg else -6.0
    prompt_style = topk_cfg.get("prompt_style", "list") if topk_cfg else "list"
    include_scores = bool(topk_cfg.get("include_scores", True)) if topk_cfg else True

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        video_inputs = video_inputs.to(device)
        video_inputs = video_inputs.float()

        logits, feats = model(video_inputs)
        log_probs = F.log_softmax(logits, dim=-1)

        if top_k <= 1:
            # ---------- original one-best path (unchanged output shape) ----------
            greedy = greedy_decode(log_probs, id_to_phoneme=reverse_vocab, blank_id=blank_id)
            predicted_phonemes = [h.phonemes for h in greedy]
            return predicted_phonemes, feats

        # ---------- top-k path ----------
        batch_hyps = topk_beam_decode(
            log_probs,
            id_to_phoneme=reverse_vocab,
            blank_id=blank_id,
            beam_width=ctc_beam_width,
            top_k=top_k,
            prune_log_prob=prune_log_prob,
        )
        prompts = [
            format_prompt_for_llm(
                hyps,
                top_k=top_k,
                include_scores=include_scores,
                style=prompt_style,
            )
            for hyps in batch_hyps
        ]
        return batch_hyps, feats, prompts

def main(args):
    mode = args.mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_path = args.videos_root
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    save_model_path = args.save_model_path
    sample_size = args.sample_size
    version = args.version
    vocab = get_vocab()

    encoder_type = getattr(args, "encoder_type", "adapter")
    multiscale_cfg = getattr(args, "multiscale_config", None)

    if mode == "train":
        print(f"Training (encoder_type={encoder_type})")
        train(
            device,
            version,
            video_path,
            batch_size,
            num_workers,
            epochs,
            save_model_path,
            sample_size,
            vocab,
            encoder_type=encoder_type,
            multiscale_config_path=multiscale_cfg,
        )
    elif mode == "infer":
        top_k = getattr(args, "ctc_top_k", 1)
        ctc_beam_width = getattr(args, "ctc_beam_width", 16)
        topk_cfg = getattr(args, "topk_config", None)
        result = run_inference(
            save_model_path,
            version,
            video_path,
            device,
            vocab,
            top_k=top_k,
            ctc_beam_width=ctc_beam_width,
            topk_config_path=topk_cfg,
            encoder_type=encoder_type,
            multiscale_config_path=multiscale_cfg,
        )
        print("Inferences", result)

if __name__ == "__main__":
    main(args)
