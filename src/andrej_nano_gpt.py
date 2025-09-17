import math

import torch
from torch import nn
from torch.functional import F

# hyperparameters
batch_size = 8  # how many independent sequences will we process in parallel?
block_size = 24  # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 64
n_head = 4
head_size = n_embed // n_head
n_layer = 8
dropout = 0.2
# ------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O dataset.txt
# with open("dataset.txt", 'r', encoding="utf-8") as f:
#     text = f.read()
with open('../data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
print("data.shape: ", data.shape)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ A Head of self-attention"""

    def __init__(self, block_size, n_embed, head_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size)
        self.key = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # compute attention scores (affinities) from the formula in the Attention is all you need paper
        wei = (q @ k.transpose(-2, -1)) / math.sqrt(
            self.block_size)  # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -float('inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        head_out = wei @ v  # (B, T, T) @ (B, T, head_size) = (B, T, head_size)

        return head_out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, block_size, n_embed, n_head, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([Head(block_size, n_embed, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ a single linear layer followed by a nonlinearity"""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(block_size, n_embed, n_head, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_size = self.n_embed // self.n_head
        self.dropout = dropout
        self.device = device

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        self.sa_heads = MultiHeadAttention(self.block_size, self.n_embed, self.n_head,
                                           self.dropout)  # sa = self-attention
        self.blocks = nn.Sequential(*[Block(self.n_embed, self.n_head, self.dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size)  # lm = Language Modelling

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T)

        tok_emb = self.token_embedding_table(idx)  # (B, T, C(n_embed))
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)  # (4, 8, 65) -> (32, 65)
        targets = targets.view(B * T)  # (32, 1)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size, block_size, n_embed, n_head, n_layer, dropout, device)
m = model.to(device)
n_params = sum([p.numel() for p in model.parameters()])
print(f"Number of parameters: {n_params}")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))


