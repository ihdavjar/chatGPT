import torch
import torch.nn as nn
import torch.nn.functional as F


#Importing the Data
text = open('/media/ihdav/Files/Research/Infrrd_Research/chatGPT/Data/Tiny_Shakespeare/input.txt','r').read()
text = open('/media/ihdav/Files/Research/Infrrd_Research/chatGPT/out_data_full.txt','r').read()

#hyperparameters
batch_size = 64 #number of sequences in a batch (B)
block_size = 128 #number of characters in a sequence (T)
max_iters = 10000
eval_intervals = 300
learning_rate = 3e-4
eval_ites = 100
n_embed = 128
num_heads = 4  
num_blocks = 6
dropout = 0.2

torch.manual_seed(1337)


device = "cuda" if torch.cuda.is_available() else "cpu"   

#Creating the tokenisers
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

print(f"Vocab Size: {vocab_size}")

stoi = {s:i for i,s in enumerate(vocab)} # String to Index
itos = {i:s for s,i in stoi.items()} # Index to String

encode = lambda str_: [stoi[char] for char in str_] #Converts a string to a list of integers
decode = lambda arr_: "".join([itos[i] for i in arr_]) #Converts a list of integers to a string

#Train and Test Split
data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_data = data[:int(0.9*len(data))] #90% of the data is used for training
test_data = data[int(0.9*len(data)):]

#Creating the dataloader
def get_batch(split):
    data = train_data if split == "train" else test_data
    start_idx = torch.randint(0, data.size(0) - block_size, (batch_size,))

    xb = torch.stack([data[idx:idx+block_size] for idx in start_idx])
    yb = torch.stack([data[idx+1:idx+block_size+1] for idx in start_idx])
    xb,yb = xb.to(device), yb.to(device)
    return xb, yb

#Evaluating the model
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "test"]:
        losses = torch.zeros(eval_ites, device=device)
        for i in range(eval_ites):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()

    return out

#Creating the Head Layer
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size,bias=False)
        self.query = nn.Linear(n_embed, head_size,bias=False)
        self.value = nn.Linear(n_embed, head_size,bias=False)
        self.out_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(head_size, head_size)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # self.dn = nn.Dropout(dropout)

    def forward(self, x):
        #x -> (B,T,n_embed) -> (32,8,32)
        B,T,C = x.shape
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        wei = queries @ keys.transpose(-2,-1)*(C**(-0.5)) #(B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        
        #For Decoder Block
        wei = wei.masked_fill(self.tril[:T,:T]==0, float("-inf")) #(B,T,T) -> (B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)


        out = wei @ values #(B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        out = self.out_proj(out) #(B,T,head_size) -> (B,T,head_size)
        out = self.out_drop(out)

        return out
    

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dn = nn.Dropout(dropout)
    
    def forward(self,x):    
        out = torch.cat([head(x) for head in self.heads], dim=-1) #(B,T,head_size*num_heads)
        out = self.proj(out)
        out = self.dn(out)
        return out #(B,T,n_embed)
    

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(n_embed, 4*n_embed),
        nn.ReLU(),
        nn.Linear(4*n_embed, n_embed),
        nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self,n_embed,num_heads) -> None:
        super().__init__()
        head_size = n_embed//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


#Creating the bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.positional = nn.Embedding(block_size,n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed,num_heads) for _ in range(num_blocks)])
        self.ln_f=  nn.LayerNorm(n_embed)
        self.nl_head = nn.Linear(n_embed, vocab_size)

    
    def forward(self, idx, target=None):
        
        #idx -> (B,T) -> (32,8)
        tok_emb = self.embedding(idx) #(B,T,n_embed) -> (32,8,65)
        pos_emb = self.positional(torch.arange(min(block_size,idx.shape[1]))) #(T,n_embed) -> (8,65)
        x = tok_emb+pos_emb #(B,T,n_embed) -> (32,8,65)

        x = self.blocks(x)
        logits = self.nl_head(self.ln_f(x)) #(B,T,vocab_size) -> (32,8,65)

        if (target==None):
            loss = None
        
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))

        return logits,loss
    
    def generate(self, idx_in, max_new_tokens):

        for i in range(max_new_tokens):

            #Cropping the inputs:
            idx_cond = idx_in[:,-block_size:]

            logits,loss = self.forward(idx_cond) #(B1,T1,C)x

            logits = logits[:,-1,:]

            #Using the exp and normalising along the C dimenstion to get the probabilities
            probs = F.softmax(logits, dim=-1) #(B1,T1,C)

            #Taking the last token probabilities (context = 1)
            # probs = probs[:,-1,:]

            #Sampling from the probabilities to get the next token
            next_token = torch.multinomial(probs, num_samples=1) #(B1,1)

            idx_in = torch.cat([idx_in, next_token], dim=1) #(B1,T1+1)

        return idx_in



#Training the model
model = BigramLanguageModel().to(device)
# model = torch.load("/media/ihdav/Files/Research/Infrrd_Research/chatGPT/BigramModel_Chats.pt")

optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    
    #getting the batch
    xb, yb = get_batch("train")
    
    #forward pass
    logits, loss = model(xb, yb)

    #backward pass
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    

    #THIS LOSS in pretty noisy as this is on each of the batch.
    #Hence we made new average kind of loss on both train and test dataset
    if (i%eval_intervals == 0):
        temp_dict = estimate_loss()
        print(f"{i}th Iteration=>\nTrain Loss: {temp_dict['train']}, Test Loss: {temp_dict['test']}")

torch.save(model, "/media/ihdav/Files/Research/Infrrd_Research/chatGPT/BigramModel_Chats.pt")
# model = BigramLanguageModel().to(device)
# model = torch.load("/media/ihdav/Files/Research/Infrrd_Research/chatGPT/BigramModel.pt")

#Context Generation
print(torch.tensor(encode("I am IhdavJar"), device=device))
context = torch.tensor([[1]], device=device)
context = torch.tensor([encode("Ihdavjar")], device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))  