"""
Fine-tune ModernBERT (or another BERT-type model) for scalar construct measurement using reward modeling.

Based on:
1. Ouyang et al. (2022) - InstructGPT reward modeling approach
2. Licht et al. (2025) - Scalar construct measurement with LLMs

This implementation trains a reward model on pairwise comparison data,
then uses it to score individual text items on a continuous scale.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import json


class RewardModel:
    """
    Unified class for training and using a reward model for text scoring.
    
    This class handles:
    - Model initialization and configuration
    - Dataset creation and management
    - Training loop with pairwise comparisons
    - Scoring individual texts or batches
    - Score normalization
    """
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-large",
        dropout: float = 0.1,
        max_length: int = 384,
        device: Optional[str] = None
    ):
        """
        Initialize the reward model trainer.
        
        Args:
            model_name: HuggingFace model identifier
            dropout: Dropout rate for reward head
            max_length: Maximum sequence length for tokenization
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = self._build_model(dropout)
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
    
    def _build_model(self, dropout: float) -> nn.Module:
        """Build the reward model architecture."""
        encoder = AutoModel.from_pretrained(self.model_name)
        hidden_size = encoder.config.hidden_size
        
        # Create complete model
        model = nn.Module()
        model.encoder = encoder
        model.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Add forward method
        def forward(input_ids, attention_mask):
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            rewards = model.reward_head(pooled_output).squeeze(-1)
            return rewards
        
        model.forward = forward
        return model
    
    def prepare_data(
        self,
        pairs: List[Tuple[str, str, Union[int, float]]],
        batch_size: int = 16,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Prepare training data from pairwise comparisons.
        
        Args:
            pairs: List of (text_winner, text_loser, margin) tuples
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for training
        """
        dataset = self._PairwiseDataset(pairs, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    class _PairwiseDataset(Dataset):
        """Internal dataset class for pairwise comparisons."""
        
        def __init__(self, pairs, tokenizer, max_length):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            text_winner, text_loser, margin = self.pairs[idx]
            
            encoding_winner = self.tokenizer(
                text_winner,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            encoding_loser = self.tokenizer(
                text_loser,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids_winner': encoding_winner['input_ids'].squeeze(0),
                'attention_mask_winner': encoding_winner['attention_mask'].squeeze(0),
                'input_ids_loser': encoding_loser['input_ids'].squeeze(0),
                'attention_mask_loser': encoding_loser['attention_mask'].squeeze(0),
                'margin': torch.tensor(margin, dtype=torch.float)
            }
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        eval_loader: Optional[DataLoader] = None,
        log_interval: int = 50
    ):
        """
        Train the reward model on pairwise comparison data.
        
        Args:
            train_loader: DataLoader with training pairs
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for scheduler
            eval_loader: Optional DataLoader for evaluation
            log_interval: Log metrics every N steps
        """
        self.model.train()
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                loss = self._training_step(batch)
                epoch_loss += loss
                
                if (step + 1) % log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.training_history.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})
            
            print(f"Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f}")
            
            # Evaluation
            if eval_loader:
                eval_loss = self.evaluate(eval_loader)
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")
                self.training_history[-1]['eval_loss'] = eval_loss
    
    def _training_step(self, batch) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # Get rewards for winner and loser
        reward_winner = self.model(
            batch['input_ids_winner'],
            batch['attention_mask_winner']
        )
        reward_loser = self.model(
            batch['input_ids_loser'],
            batch['attention_mask_loser']
        )
        
        # Pairwise ranking loss (winner should have higher reward)
        loss = -torch.log(torch.sigmoid(reward_winner - reward_loser)).mean()
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate the model on a validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                reward_winner = self.model(
                    batch['input_ids_winner'],
                    batch['attention_mask_winner']
                )
                reward_loser = self.model(
                    batch['input_ids_loser'],
                    batch['attention_mask_loser']
                )
                
                loss = -torch.log(torch.sigmoid(reward_winner - reward_loser)).mean()
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(eval_loader)
    
    def score_text(self, text: str) -> float:
        """
        Score a single text item.
        
        Args:
            text: Text to score
            
        Returns:
            Scalar reward score
        """
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            reward = self.model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device)
            )
        
        return reward.item()
    
    def score_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Score multiple texts efficiently.
        
        Args:
            texts: List of texts to score
            batch_size: Batch size for processing
            
        Returns:
            Array of scores
        """
        self.model.eval()
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                rewards = self.model(
                    encodings['input_ids'].to(self.device),
                    encodings['attention_mask'].to(self.device)
                )
            
            scores.extend(rewards.cpu().numpy())
        
        return np.array(scores)
    
    def normalize_scores(
        self,
        scores: np.ndarray,
        scale_min: float = 1.0,
        scale_max: float = 9.0
    ) -> np.ndarray:
        """
        Normalize raw reward scores to a desired scale.
        
        Args:
            scores: Raw scores to normalize
            scale_min: Minimum value of output scale
            scale_max: Maximum value of output scale
            
        Returns:
            Normalized scores
        """
        score_min = scores.min()
        score_max = scores.max()
        
        if score_max == score_min:
            return np.full_like(scores, (scale_min + scale_max) / 2)
        
        normalized = (scores - score_min) / (score_max - score_min)
        normalized = normalized * (scale_max - scale_min) + scale_min
        
        return normalized
    
    def save(self, path: str):
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'config': {
                'model_name': self.model_name,
                'max_length': self.max_length
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', [])
        print(f"Model loaded from {path}")