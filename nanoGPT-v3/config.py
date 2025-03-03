import os
from datetime import datetime
import pickle

class Config:
    def __init__(self):
        # I/O
        self.out_dir = 'runs'
        self.run_dir = None
        self.eval_interval = 100
        self.log_interval = 1
        self.eval_iters = 100
        self.eval_only = False
        self.always_save_checkpoint = False
        self.init_from = 'scratch'
        
        # Tokenizer tracking
        self.tokenizer_path = None  # Will be set based on dataset
        self.tokenizer_name = None  # Will be extracted from path
        
        # data
        self.dataset = 'seq_512'
        self.gradient_accumulation_steps = 8
        self.batch_size = 32
        self.block_size = 512
        
        # model
        self.n_layer = 3
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.1
        self.bias = False

        # MLA
        self.rope_theta = 10000.0

        # adamw optimizer
        self.learning_rate = 3e-4
        self.max_iters = 150000
        self.weight_decay = 0.03
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # learning rate decay settings
        self.decay_lr = True
        self.warmup_iters = 1000
        self.lr_decay_iters = 150000
        self.min_lr = 6e-5

        # Mixture of Experts
        self.n_shared_experts = 2
        self.n_routed_experts = 2
        self.top_k_experts = 1
        self.bias_update_speed = 0.01
        self.balance_factor = 0.01
        
        # system
        self.device = 'cuda'
        self.dtype = 'float16'
        self.compile = False

    def setup_tokenizer_from_data_dir(self, data_dir):
        """Extract and set tokenizer information from the data directory's meta.pkl"""
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self.meta = meta  # Store the entire meta dictionary
                
                # Extract tokenizer info
                if 'tokenizer_path' in meta:
                    self.tokenizer_path = meta['tokenizer_path']
                    self.tokenizer_name = os.path.basename(self.tokenizer_path)
                    print(f"Loaded tokenizer info from meta.pkl: {self.tokenizer_name}")
                    
                    # Also store vocab size from meta
                    if 'vocab_size' in meta:
                        self.vocab_size = meta['vocab_size']
                else:
                    print("Warning: meta.pkl exists but doesn't contain tokenizer_path")
            return True
        else:
            print(f"Warning: No meta.pkl found in {data_dir}")
            return False

    def setup_run_dir(self, model_name="gpt-moe"):
        """Create a unique run directory based on timestamp and tokenizer info"""
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Include tokenizer name in run directory if available
        if self.tokenizer_name:
            tokenizer_prefix = self.tokenizer_name.replace('.json', '')
            self.run_dir = os.path.join(self.out_dir, f"{model_name}_{tokenizer_prefix}_{current_time}")
        else:
            self.run_dir = os.path.join(self.out_dir, f"{model_name}_{current_time}")
        
        return self.run_dir

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())