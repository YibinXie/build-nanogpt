from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, Features
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import torch
from tqdm import tqdm
import random
import json
import tiktoken


class EYLSFTStaticDataset(Dataset):
    def __init__(self, block_size, split):
        super().__init__()

        if split == 'train':
            with open("../data/sft_train.json") as fp:
                data = json.load(fp)
        elif split == 'test':
            with open("../data/sft_test.json") as fp:
                data = json.load(fp)

        self.tokens = []
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']  # end of text token

        cnt = 0
        print(f'Loading {split} dataset...')
        for chosen in tqdm(data):
            cnt += 1
            response_text = chosen
            response = self.enc.encode(response_text)
            response.append(self.eot)

            self.tokens += response

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]  # next token prediction
        return x, y


if __name__ == '__main__':
    dataset = EYLSFTStaticDataset(1024, 'train')
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
