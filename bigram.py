import torch
import torch.nn as nn
import torch.nn.functional as F


#Importing the Data
text = open('/media/ihdav/Files/Research/Infrrd_Research/chatGPT/Data/Tiny_Shakespeare/input.txt','r').read()

#hyperparameters
batch_size = 32 #number of sequences in a batch (B)
block_size = 8 #number of characters in a sequence (T)
max_iters = 3000
eval_intervals = 300
learning_rate = 1e-2
eval_ites = 100

torch.manual_seed(1337)


device = "cuda" if torch.cuda.is_available() else "cpu"   

#Creating the tokenisers
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

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

#Creating the bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, target=None):
        
        #idx -> (B,T) -> (32,8)
        logits = self.embedding(idx) #(B,T,C) -> (32,8,65)

        if (target==None):
            loss = None
        
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))

        return logits,loss
    
    def generate(self, idx_in, max_new_tokens):

        for i in range(max_new_tokens):
            logits = self.embedding(idx_in) #(B1,T1,C)

            #Using the exp and normalising along the C dimenstion to get the probabilities
            probs = F.softmax(logits, dim=-1) #(B1,T1,C)

            #Taking the last token probabilities (context = 1)
            probs = probs[:,-1,:]

            #Sampling from the probabilities to get the next token
            next_token = torch.multinomial(probs, num_samples=1) #(B1,1)

            idx_in = torch.cat([idx_in, next_token], dim=1) #(B1,T1+1)

        return idx_in

#Training the model
model = BigramLanguageModel(vocab_size).to(device)

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

#Context Generation
context = torch.tensor([[0],[1]], device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))  