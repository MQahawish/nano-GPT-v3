import os
import torch
import random
import argparse
from contextlib import nullcontext
import json
from model import GPTConfig, GPT
from encoding_decoding import Decoder

# Global defaults (overridable via argparse)
out_dir = os.path.join('models', '3layers12heads32batch512seq-simple')
tokenizer_path = os.path.join('nanoGPT-v3', 'tokenizers', 'simple_tokenizer.json')
num_samples = 5
temperature = 0.9
top_k = 100
max_tokens = 512     
seed = random.randint(0, 1000000)
compile_model = False

midi_samples_dir = "midi-samples"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model():
    print(f"Loading model from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)
    return model

class SimpleTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.itos = {int(k): v for k, v in data['itos'].items()}
        self.stoi = data['stoi']
        self.vocab_size = data['vocab_size']
        self.start_token = "START"
        self.pad_token = "[PAD]"
        self.start_token_id = self.stoi.get(self.start_token, 0)
        self.pad_token_id = self.stoi.get(self.pad_token, self.vocab_size - 1)
    
    def encode(self, text, add_special_tokens=False):
        tokens = text.split()
        token_ids = []
        for token in tokens:
            if token in self.stoi:
                token_ids.append(self.stoi[token])
            else:
                token_ids.append(self.pad_token_id)
                print(f"Warning: unknown token '{token}' found in text")
        if add_special_tokens:
            token_ids = [self.start_token_id] + token_ids
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        tokens = []
        for idx in token_ids:
            if skip_special_tokens and idx in {self.start_token_id, self.pad_token_id}:
                continue
            tokens.append(self.itos.get(idx, self.pad_token))
        return " ".join(tokens)

def load_tokenizer():
    print(f"Loading tokenizer from {tokenizer_path}")
    return SimpleTokenizer(tokenizer_path)

def get_initial_sequence(tokenizer):
    # Return an empty initial sequence so that add_special_tokens adds the "START" token automatically.
    return "sepxx d3", "silence_first"



def generate_music_sequence_stream(model, tokenizer, initial_prompt, temperature=temperature, top_k=top_k, max_tokens=max_tokens):
    init_ids = tokenizer.encode(initial_prompt, add_special_tokens=True)
    full_sequence = init_ids.copy()
    print(f"Starting generation with {len(full_sequence)} initial tokens")
    print(f"Will generate exactly {max_tokens} tokens")
    initial_text = tokenizer.decode(full_sequence, skip_special_tokens=False)
    yield initial_text, initial_text
    tokens_generated = 0
    context_window = 512
    while tokens_generated < max_tokens:
        context = full_sequence if len(full_sequence) < context_window else full_sequence[-(context_window - 1):]
        input_ids = torch.tensor([context], dtype=torch.long).to(device)
        with torch.no_grad():
            with ctx:
                logits, _ = model(input_ids)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()
        full_sequence.append(next_token_id)
        tokens_generated += 1
        new_token = tokenizer.decode([next_token_id], skip_special_tokens=False)
        full_text = tokenizer.decode(full_sequence, skip_special_tokens=False)
        yield new_token, full_text

def generate_and_save_midi_streaming(model, tokenizer, input_text, output_midi_path, temperature=temperature, top_k=top_k, max_tokens=max_tokens, tempo=120.0):
    full_sequence = ""
    for new_token, full_text in generate_music_sequence_stream(model, tokenizer, input_text, temperature, top_k, max_tokens):
        full_sequence = full_text
        print(new_token, end='', flush=True)
    print("\nGeneration complete!")
    clean_sequence = " ".join(full_sequence.split())
    decoder = Decoder()
    decoder.text_to_midi(text=clean_sequence,
                         output_dir=os.path.dirname(output_midi_path),
                         name=os.path.basename(output_midi_path).replace('.mid', ''),
                         bpm=tempo)
    return clean_sequence

def main(num_samples):
    model = load_model()
    tokenizer = load_tokenizer()
    existing_midi_files = len([f for f in os.listdir(midi_samples_dir) if f.endswith('.mid')])
    start_index = existing_midi_files
    print(f"Found {existing_midi_files} MIDI files")
    print(f"Starting generation from index {start_index + 1}")
    for i in range(num_samples):
        current_index = start_index + i + 1
        initial_sequence, file_source = get_initial_sequence(tokenizer)
        print(f"\nGenerating sample {current_index} using sequence from {file_source}")
        print(f"Initial sequence: {initial_sequence}")
        try:
            midi_output_path = os.path.join(midi_samples_dir, f"sample_{current_index}.mid")
            generate_and_save_midi_streaming(
                model,
                tokenizer,
                initial_sequence,
                output_midi_path=midi_output_path,
                tempo=args.tempo
            )
            print(f"MIDI file created at: {midi_output_path}")
        except Exception as e:
            print(f"Error generating sample {current_index}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music sequence in custom mode only.")
    parser.add_argument('--out_dir', type=str, default=os.path.join('models', '3layers12heads32batch512seq-simple'),
                        help="Directory for model checkpoint")
    parser.add_argument('--tokenizer_path', type=str, default=os.path.join('nanoGPT-v3', 'tokenizers', 'simple_tokenizer.json'),
                        help="Path to the tokenizer JSON file")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of samples to generate")
    parser.add_argument('--temperature', type=float, default=0.9, help="Temperature for generation")
    parser.add_argument('--top_k', type=int, default=100, help="Top K tokens for sampling")
    parser.add_argument('--max_tokens', type=int, default=512, help="Fixed tokens for streaming generation")
    parser.add_argument('--seed', type=int, default=None, help="Random seed (if not provided, one is generated)")
    parser.add_argument('--compile_model', action='store_true', help="Compile model with torch.compile")
    parser.add_argument('--midi_samples_dir', type=str, default="midi-samples", help="Directory to save MIDI samples")
    parser.add_argument('--tempo', type=float, default=120.0, help="Tempo for MIDI conversion")
    
    args = parser.parse_args()
    
    out_dir = args.out_dir
    tokenizer_path = args.tokenizer_path
    num_samples = args.num_samples
    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens
    seed = args.seed if args.seed is not None else random.randint(0, 1000000)
    compile_model = args.compile_model
    midi_samples_dir = args.midi_samples_dir
    
    os.makedirs(midi_samples_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    try:
        main(num_samples)
        print("Generation and MIDI conversion complete!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
