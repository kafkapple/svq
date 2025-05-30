import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from src.models.svq_model import SVQ
from src.data.toy_datasets import get_data_loaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        dataset_name=cfg.experiment.dataset.name,
        batch_size=cfg.experiment.dataset.batch_size,
        num_workers=cfg.experiment.dataset.num_workers,
        num_samples=cfg.experiment.dataset.num_samples,
        image_size=cfg.experiment.dataset.image_size,
        max_shapes=cfg.experiment.dataset.max_shapes
    )
    
    # Create model
    model = SVQ(
        image_size=cfg.experiment.dataset.image_size,
        in_channels=3,
        num_slots=cfg.experiment.model.num_slots,
        num_iterations=cfg.experiment.model.num_iterations,
        slot_size=cfg.experiment.model.slot_size,
        num_codebooks=cfg.experiment.model.num_codebooks,
        codebook_size=cfg.experiment.model.codebook_size,
        code_dim=cfg.experiment.model.code_dim,
        hidden_dim=cfg.experiment.model.hidden_dim,
        commitment_cost=cfg.experiment.model.commitment_cost
    ).to(device)
    
    # Initialize autoregressive prior if enabled
    if cfg.experiment.model.use_prior:
        model.init_prior(
            embed_dim=cfg.experiment.model.prior.embed_dim,
            num_heads=cfg.experiment.model.prior.num_heads,
            num_layers=cfg.experiment.model.prior.num_layers,
            dropout=cfg.experiment.model.prior.dropout
        )
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.experiment.training.learning_rate)
    
    # Training loop
    for epoch in range(cfg.experiment.training.num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            loss, loss_dict = model.compute_loss(
                data,
                outputs,
                recon_loss_weight=cfg.experiment.training.recon_loss_weight,
                commitment_loss_weight=cfg.experiment.training.commitment_loss_weight
            )
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
                for k, v in loss_dict.items():
                    logger.info(f"  {k}: {v:.4f}")
        
        avg_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Evaluation
        if epoch % cfg.experiment.training.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    outputs = model(data)
                    loss, _ = model.compute_loss(
                        data,
                        outputs,
                        recon_loss_weight=cfg.experiment.training.recon_loss_weight,
                        commitment_loss_weight=cfg.experiment.training.commitment_loss_weight
                    )
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % cfg.experiment.training.save_interval == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
