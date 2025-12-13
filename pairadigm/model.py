"""
Fine-tune ModernBERT (or another BERT-type model) for scalar construct measurement using reward modeling.

Based on:
1. Ouyang et al. (2022) - InstructGPT reward modeling approach
2. Licht et al. (2025) - Scalar construct measurement with LLMs

This implementation trains a reward model on pairwise comparison data,
then uses it to score individual text items on a continuous scale.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from pairadigm import Pairadigm
import copy


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
        device: Optional[str] = None,
        Pairadigm: Optional['Pairadigm'] = None
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
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Model using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = self._build_model(dropout)
        self.model.to(self.device)

        # If a Pairadigm instance is provided, link it
        self.pairadigm = Pairadigm
        
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
        pairs: Optional[List[Tuple[str, str, Union[int, float]]]] = None,
        batch_size: int = 16,
        shuffle: bool = True,
        decision_col: str = "decision",
        score_col: str = "Bradley_Terry_Score",
        margins: bool = True,
        test_size: float = 0.15,
        eval_size: float = 0.15,
        stratify: Optional[List] = None,
        random_state: Optional[int] = 42
    ) -> Dict[str, DataLoader]:
        """
        Prepare training, evaluation, and test data from pairwise comparisons.

        Args:
            pairs: Optional list of (text_winner, text_loser, margin) tuples. If None and a Pairadigm instance is linked, will use Pairadigm.
            batch_size: Batch size for training
            shuffle: Whether to shuffle the training data
            decision_col: Column name for decision in pairwise_df
            score_col: Column name for scores in scored_df
            margins: Whether to compute margins from scores
            test_size: Proportion of data for test set (default 0.15)
            eval_size: Proportion of data for eval set (default 0.15)
            stratify: Optional list of labels for stratified splitting
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'eval', and 'test' DataLoaders
        """
        if pairs is None and self.pairadigm:
            pairs = self._prepare_pairadigm(
                decision_col=decision_col, 
                score_col=score_col, 
                margins=margins
            )
        elif pairs is None:
            raise ValueError("No pairs provided and no Pairadigm instance linked.")
        
        # Split data: first split into train+eval and test
        train_eval_pairs, test_pairs = train_test_split(
            pairs,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state
        )
        
        # Then split train+eval into train and eval
        # Adjust eval_size to be relative to train+eval size
        adjusted_eval_size = eval_size / (1 - test_size)
        train_pairs, eval_pairs = train_test_split(
            train_eval_pairs,
            test_size=adjusted_eval_size,
            stratify=None,  # stratify only on first split
            random_state=random_state
        )
        
        print(f"Data split - Train: {len(train_pairs)}, Eval: {len(eval_pairs)}, Test: {len(test_pairs)}")
        
        # Create datasets and dataloaders
        train_dataset = self._PairwiseDataset(train_pairs, self.tokenizer, self.max_length)
        eval_dataset = self._PairwiseDataset(eval_pairs, self.tokenizer, self.max_length)
        test_dataset = self._PairwiseDataset(test_pairs, self.tokenizer, self.max_length)
        
        return {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
            'eval': DataLoader(eval_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
    
    def _prepare_pairadigm(
            self,
            decision_col: str = "decision", 
            score_col: str = "Bradley_Terry_Score",
            margins: bool = True):
        """
        Prepare pairwise data from linked Pairadigm instance.
        Args:
            decision_col: Column in pairwise_df indicating winner/loser
            score_col: Column in scored_df with precomputed scores
            margins: Whether to compute margins from scores
        Returns:
            Tuples with pairs and margins
        """
        if self.pairadigm is None:
            raise ValueError("No Pairadigm instance linked to RewardModel.")
        if decision_col not in self.pairadigm.pairwise_df.columns:
            raise ValueError(f"Decision column '{decision_col}' not found in pairwise_df. Run generate_pairwise_annotations() first.")
        if "breakdown1" not in self.pairadigm.pairwise_df.columns or "breakdown2" not in self.pairadigm.pairwise_df.columns:
            raise ValueError("Breakdown columns not found in pairwise_df. Run generate_breakdowns() and generate_pairings(breakdowns=True) first.")

        pairwise_df = self.pairadigm.pairwise_df.copy()

        original_text = self.pairadigm.data[[self.pairadigm.item_id_name, 
                                             self.pairadigm.text_name]].copy()
        
        pairwise_df = pd.merge(
            pairwise_df, original_text,
            left_on='item1',
            right_on=self.pairadigm.item_id_name,
            how='left').rename(columns={self.pairadigm.text_name: 'text1'}).drop(columns=[self.pairadigm.item_id_name]).merge(
                original_text,
                left_on='item2',
                right_on=self.pairadigm.item_id_name,
                how='left'
                ).rename(columns={self.pairadigm.text_name: 'text2'}).drop(columns=[self.pairadigm.item_id_name])
        
        # Calculate margins when desired, if scores are available
        if margins: 
            if not self.pairadigm.scored_df:
                raise ValueError("No scored_df found in linked Pairadigm instance.")
            if score_col not in self.pairadigm.scored_df.columns:
                raise ValueError(f"Score column '{score_col}' not found in scored_df. Run score_items() first.")

            pairs = pd.merge(
                pairwise_df[['item1', 'item2', 
                             'text1', 'text2', decision_col]],
                self.pairadigm.scored_df[[self.pairadigm.item_id_name, score_col]],
                    left_on='item1',
                    right_on=self.pairadigm.item_id_name,
                    how='left'
                ).rename(columns={score_col: 'item_A_score'}).drop(columns=[self.pairadigm.item_id_name]).merge(
                    self.pairadigm.scored_df[[self.pairadigm.item_id_name, score_col]],
                    left_on='item2',
                    right_on=self.pairadigm.item_id_name,
                    how='left'
                ).rename(columns={score_col: 'item_B_score'}).drop(columns=[self.pairadigm.item_id_name])
            
            pairs['margin'] = pairs['item_B_score'] - pairs['item_A_score']

            # Make a tuple of breakdown1, breakdown2, margin for each row in prepped_data
            training_pairs = list(zip(
                pairs['text1'],
                pairs['text2'],
                pairs['margin']
            ))

            return training_pairs
        
        # If margins not desired, just return winner/loser pairs
        training_pairs = []
        for _, row in self.pairadigm.pairwise_df.iterrows():
            if row[decision_col] == 1 or row[decision_col] == 'Text1':
                training_pairs.append((row['text1'], row['text2'], 1.0))
            elif row[decision_col] == 2 or row[decision_col] == 'Text2':
                training_pairs.append((row['text2'], row['text1'], 1.0))
            else:
                continue  # Skip ties or invalid decisions

        return training_pairs
    
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
        eval_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        log_interval: int = 50,
        early_stopping_patience: int = 3
    ):
        """
        Train the reward model on pairwise comparison data with optional early stopping.

        Args:
            train_loader: DataLoader with training pairs
            eval_loader: DataLoader for evaluation (required for early stopping)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for scheduler
            log_interval: Log metrics every N steps
            early_stopping_patience: Number of epochs with no improvement on eval loss before stopping early.
                                     Set to None or 0 to disable early stopping.
        Returns:
            The model (self.model) restored to the best-performing weights observed on eval data.
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

        # Early stopping bookkeeping
        best_state = None
        best_eval_loss = float('inf')
        epochs_without_improve = 0
        use_early_stopping = bool(early_stopping_patience and eval_loader is not None and early_stopping_patience > 0)
        
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
            epoch_metrics = {'epoch': epoch + 1, 'train_loss': avg_epoch_loss}
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f}")
            
            # Evaluation
            if eval_loader:
                eval_metrics = self.evaluate(eval_loader)
                epoch_metrics.update(eval_metrics)
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_metrics['eval_loss']:.4f}, Eval Accuracy: {eval_metrics['eval_accuracy']:.4f}")
                
                # Check for improvement on eval_loss and save best model
                current_eval_loss = eval_metrics.get('eval_loss', float('inf'))
                if current_eval_loss < best_eval_loss:
                    best_eval_loss = current_eval_loss
                    # store a CPU copy of the state dict
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improve = 0
                    print(f"  New best model found (eval_loss improved to {best_eval_loss:.4f}).")
                else:
                    epochs_without_improve += 1
                    print(f"  No improvement for {epochs_without_improve} epoch(s).")
            
            else:
                # If no eval_loader provided, we can't do early stopping / track best by eval
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            self.training_history.append(epoch_metrics)

            # Early stopping check
            if use_early_stopping and epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs (no improvement in eval loss for {early_stopping_patience} epochs).")
                break
        
        # Restore best model weights if we tracked them
        if best_state is not None:
            # move tensors back to device as needed when loading
            device_state = {k: v.to(self.device) for k, v in best_state.items()}
            self.model.load_state_dict(device_state)
            print("Best model weights restored based on eval data.")
        
        return self.model
    
    def _training_step(self, batch) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Move batch to device - fix the iteration
        batch = {key: value.to(self.device) for key, value in batch.items()}
        
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
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a validation set.
        
        Args:
            eval_loader: DataLoader with evaluation pairs
            
        Returns:
            Dictionary with 'eval_loss' and 'eval_accuracy' metrics
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                # Fix the iteration here too
                batch = {key: value.to(self.device) for key, value in batch.items()}
                
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
                
                # Calculate accuracy (winner should have higher reward)
                correct_predictions += (reward_winner > reward_loser).sum().item()
                total_predictions += len(reward_winner)
        
        self.model.train()
        return {
            'eval_loss': total_loss / len(eval_loader),
            'eval_accuracy': correct_predictions / total_predictions
        }
    
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
        scale_min: float = 0.0,
        scale_max: float = 1.0
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
    
    def test_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on the test set and print detailed results.
        
        Args:
            test_loader: DataLoader with test pairs
            
        Returns:
            Dictionary with test metrics
        """
        print("\n" + "="*50)
        print("Running Test Evaluation")
        print("="*50)
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_margins = []
        all_predicted_margins = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                
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
                
                # Calculate accuracy
                correct_predictions += (reward_winner > reward_loser).sum().item()
                total_predictions += len(reward_winner)
                
                # Track margins for correlation analysis
                predicted_margins = (reward_winner - reward_loser).cpu().numpy()
                actual_margins = batch['margin'].cpu().numpy()
                all_predicted_margins.extend(predicted_margins)
                all_margins.extend(actual_margins)
        
        # Calculate metrics
        test_loss = total_loss / len(test_loader)
        test_accuracy = correct_predictions / total_predictions
        
        # Calculate correlation between predicted and actual margins
        correlation = np.corrcoef(all_margins, all_predicted_margins)[0, 1]
        
        # Print results
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"  Margin Correlation: {correlation:.4f}")
        print("\n" + "="*50 + "\n")
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'margin_correlation': correlation,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
        
        return results
    
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

    def push_to_hub(self, repo_id: str, private: bool = True):
        """
        Push the trained model to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID in format "username/repo-name"
            private: Whether to make the repository private (default True)
        """
        from huggingface_hub import create_repo, upload_folder
        import os
        
        # Create a temporary directory for the model files
        temp_dir = f"./temp_model_{repo_id.split('/')[-1]}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save encoder and reward head
            self.model.encoder.save_pretrained(os.path.join(temp_dir, "encoder"))
            self.tokenizer.save_pretrained(temp_dir)
            torch.save(self.model.reward_head.state_dict(), os.path.join(temp_dir, "reward_head.pt"))
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'training_history': self.training_history
            }
            with open(os.path.join(temp_dir, "training_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create and upload to Hub
            create_repo(repo_id, private=private, exist_ok=True)
            upload_folder(repo_name=repo_id, folder_path=temp_dir, repo_type="model")
            
            print(f"Model successfully pushed to HuggingFace Hub: https://huggingface.co/{repo_id}")
        
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)