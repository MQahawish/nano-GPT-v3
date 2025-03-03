import os
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, run_dir):
        """
        Initialize TensorBoard logger using the run directory
        
        Args:
            run_dir: Directory for this specific run
        """
        # Create tensorboard directory within run directory
        tensorboard_dir = os.path.join(run_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(tensorboard_dir)
        
    def log_training(self, iter_num, loss, lr, mfu=None):
        """Log training metrics"""
        self.writer.add_scalar('loss/train', loss, iter_num)
        self.writer.add_scalar('learning_rate', lr, iter_num)
        if mfu is not None:
            self.writer.add_scalar('mfu', mfu, iter_num)
    
    def log_evaluation(self, iter_num, val_loss):
        """Log evaluation metrics"""
        self.writer.add_scalar('loss/val', val_loss, iter_num)
        
    def log_hparams(self, config):
        """Log hyperparameters"""
        hparam_dict = {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'dropout': config.dropout,
            'weight_decay': config.weight_decay
        }
        self.writer.add_hparams(hparam_dict, {})
        
    def close(self):
        """Close the writer"""
        self.writer.close()