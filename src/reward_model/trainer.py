"""Training logic for reward model."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging

from src.reward_model.model import RewardModel
from src.utils.logging_utils import setup_logger
from src.utils.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


class JudgmentDataset(Dataset):
    """Dataset for reward model training."""

    def __init__(
        self,
        judgments: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """Initialize dataset.

        Args:
            judgments: List of judgment dictionaries
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.judgments = judgments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.judgments)

    def __getitem__(self, idx: int) -> Dict:
        item = self.judgments[idx]

        # Create input text: context + student message + tutor response
        input_text = (
            f"Context: {item['context']}\n"
            f"Student: {item['student_message']}\n"
            f"Tutor: {item['tutor_response']}"
        )

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get score (overall score from judgment)
        score = float(item['judgment']['overall']['score'])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float32)
        }


class RewardModelTrainer:
    """Train reward models from Claude judgments."""

    def __init__(self, config_path: str = "config/reward_model_config.yaml"):
        """Initialize trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def prepare_data(
        self,
        judgments_path: str
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare train/val/test datasets.

        Args:
            judgments_path: Path to judgments JSON file

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info(f"Loading judgments from {judgments_path}")

        with open(judgments_path, 'r') as f:
            judgments = json.load(f)

        logger.info(f"Loaded {len(judgments)} judgments")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['base_model']
        )

        # Split data
        np.random.seed(self.config['data']['seed'])
        indices = np.random.permutation(len(judgments))

        train_size = int(len(judgments) * self.config['data']['train_split'])
        val_size = int(len(judgments) * self.config['data']['val_split'])

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_judgments = [judgments[i] for i in train_indices]
        val_judgments = [judgments[i] for i in val_indices]
        test_judgments = [judgments[i] for i in test_indices]

        logger.info(
            f"Split: {len(train_judgments)} train, "
            f"{len(val_judgments)} val, {len(test_judgments)} test"
        )

        # Create datasets
        train_dataset = JudgmentDataset(
            train_judgments, tokenizer, self.config['model']['max_length']
        )
        val_dataset = JudgmentDataset(
            val_judgments, tokenizer, self.config['model']['max_length']
        )
        test_dataset = JudgmentDataset(
            test_judgments, tokenizer, self.config['model']['max_length']
        )

        return train_dataset, val_dataset, test_dataset

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        save_path: Optional[str] = None
    ) -> Dict:
        """Train reward model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            save_path: Path to save model (uses config default if None)

        Returns:
            Training metrics dictionary
        """
        save_path = save_path or self.config['output']['save_path']
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize model
        model = RewardModel(
            base_model_name=self.config['model']['base_model'],
            hidden_size=self.config['model']['hidden_size'],
            dropout=self.config['model']['dropout'],
            max_length=self.config['model']['max_length']
        ).to(self.device)

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        total_steps = len(train_loader) * self.config['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=total_steps
        )

        # Loss function (MSE for regression)
        criterion = torch.nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []

        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_loss = self._train_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )

            # Validate
            val_loss, val_metrics = self._validate(
                model, val_loader, criterion
            )

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_pearson={val_metrics['pearson']:.4f}"
            )

            serializable_metrics = {
                key: float(value)
                for key, value in val_metrics.items()
            }
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                **serializable_metrics
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model(model, save_path)
                logger.info(f"✓ Saved new best model with val_loss={val_loss:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if (self.config['training']['early_stopping']['enabled'] and
                patience_counter >= self.config['training']['early_stopping']['patience']):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        # Save training history
        history_path = Path(save_path) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'history': training_history
        }

    def _train_epoch(
        self,
        model,
        dataloader,
        optimizer,
        scheduler,
        criterion
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            scores = batch['score'].to(self.device)

            # Forward pass
            predictions = model(input_ids, attention_mask).squeeze(-1)
            loss = criterion(predictions, scores)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config['training']['max_grad_norm']
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _validate(self, model, dataloader, criterion) -> Tuple[float, Dict]:
        """Validate model."""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                scores = batch['score'].to(self.device)

                predictions = model(input_ids, attention_mask).squeeze(-1)
                loss = criterion(predictions, scores)

                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(scores.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        metrics = compute_regression_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )

        return avg_loss, metrics

    def _save_model(self, model, save_path: str) -> None:
        """Save model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(model.state_dict(), save_path / 'model.pt')

        # Save config
        with open(save_path / 'config.json', 'w') as f:
            json.dump(model.get_config(), f, indent=2)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['base_model']
        )
        tokenizer.save_pretrained(save_path)
