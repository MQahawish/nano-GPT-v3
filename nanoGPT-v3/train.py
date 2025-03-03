import os
import time
import math
import pickle
from contextlib import nullcontext
import random
import numpy as np
import torch
import logging
from config import Config
from model import GPTConfig, GPT
from tensorboard_logger import TensorBoardLogger

# -----------------------------------------------------------------------------
# Initialize configuration
# -----------------------------------------------------------------------------
config = Config()
run_dir = config.setup_run_dir()
os.makedirs(run_dir, exist_ok=True)
tensorboard = TensorBoardLogger(run_dir)
tensorboard.log_hparams(config)
# -----------------------------------------------------------------------------
# Create output directory and set up logging
# -----------------------------------------------------------------------------
if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir, exist_ok=True)

# Set up logging configuration first
# Set up logging configuration first
logging.basicConfig(
    filename=os.path.join(run_dir, 'training.log'),
    filemode='a',
    format='%(name)s - %(levelname)s - %(message)s'
)

# Then set up eval logger
eval_logger = logging.getLogger('eval')
eval_handler = logging.FileHandler(os.path.join(run_dir, 'eval.log'), mode='a')
eval_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
eval_handler.setFormatter(eval_formatter)
eval_logger.addHandler(eval_handler)
eval_logger.setLevel(logging.WARNING)

# Save config with tokenizer info
config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
with open(os.path.join(run_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config_dict, f)

# Also save human-readable version
with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
    f.write(str(config))
# -----------------------------------------------------------------------------
# Initialize training parameters
# -----------------------------------------------------------------------------
config_keys = [k for k, v in config.__dict__.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
master_process = True
seed_offset = 0
tokens_per_iter = config.gradient_accumulation_steps * config.batch_size * config.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Set up device and seeds
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in config.device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
data_dir = os.path.join(r'C:\Users\97250\OneDrive\Documents\GitHub\GPT-R0\piano-binaries-fixed-stride\simple_tokenizer_aug30', config.dataset)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

# Setup tokenizer information from data directory
config.setup_tokenizer_from_data_dir(data_dir)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    xs, ys = [], []
    while len(xs) < config.batch_size:
        ix = random.randrange(0, int(len(data) - config.block_size))

        x = data[ix:ix + config.block_size]
        y = data[ix + 1:ix + 1 + config.block_size]
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        xs.append(x)
        ys.append(y)
    
    x = torch.stack(xs)
    y = torch.stack(ys)
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)    
    return x, y

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = math.inf

# Load vocabulary size from meta
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    block_size=config.block_size,
    bias=config.bias,
    vocab_size=None,
    dropout=config.dropout,
    n_shared_experts=config.n_shared_experts,
    n_routed_experts=config.n_routed_experts,
    top_k_experts=config.top_k_experts,
    bias_update_speed=config.bias_update_speed,
    balance_factor=config.balance_factor
)

# Model initialization based on config
if config.init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif config.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
    override_args = dict(dropout=config.dropout)
    model = GPT.from_pretrained(config.init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    model_args['block_size'] = config.block_size

model.to(config.device)

# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

if config.init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if config.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Training functions
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model
running_mfu = -1.0

try:
    while True:
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % config.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            eval_logger.warning(
                f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}, "
                f"lr {lr:.6f}, mfu {running_mfu * 100:.2f}%")
            tensorboard.log_evaluation(iter_num, losses['val'])
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
                    print(f"saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)
                    
                    # Optionally save best checkpoint separately
                    if losses['val'] < best_val_loss:
                        best_checkpoint_path = os.path.join(run_dir, 'best_checkpoint.pt')
                        torch.save(checkpoint, best_checkpoint_path)
                    
        if iter_num == 0 and config.eval_only:
            break

        for micro_step in range(config.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                
                # Add MoE balance loss
                moe_loss = 0
                for block in model.transformer.h:
                    moe_loss += block.mlp.get_loss()
                
                # Combine the losses
                total_loss = loss + moe_loss
                total_loss = total_loss / config.gradient_accumulation_steps
                loss = total_loss  # Keep loss variable for logging
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config.log_interval == 0 and master_process:
            lossf = loss.item() * config.gradient_accumulation_steps
            
            # Get stats once
            stats = raw_model.get_gpu_stats(config.batch_size * config.gradient_accumulation_steps, dt)
            mfu = stats['mfu']
            
            # Update running MFU if we're past warmup
            if local_iter_num >= 5:
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            
            # Print comprehensive stats
            print(f"Step {iter_num}: "
                f"loss {lossf:.4f} | "
                f"MFU: {mfu*100:.2f}% | "
                f"VRAM: {stats['memory_allocated_mb']:.0f}MB ({stats['memory_percentage']:.1f}%) | "
                f"time {dt*1000:.2f}ms")
            
            # Log to tensorboard
            tensorboard.log_training(iter_num, lossf, lr, running_mfu)

        iter_num += 1
        local_iter_num += 1

        if iter_num > config.max_iters:
            break

except Exception as e:
    print(f"Training interrupted by error: {str(e)}")
    raise

finally:
    # Close tensorboard logger
    tensorboard.close()