import os
import csv
import glob
import time
import json
import gzip
import torch
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader

max_sentence_len="<MAX_LEN>"
max_new_tokens="<MAX_NEW_TOKENS>"

class QnADataset(IterableDataset):
    '''Dataset that streams batches instead of loading the whole file into memory'''
    def __init__(self, files_path, max_sentence_len=256, shuffle=True, tokenizer=None):
        super(QnADataset).__init__()
        self.files = glob.glob(os.path.join(files_path, "*.csv.gz"))
        if len(self.files) == 0: raise Exception("No .csv files found")
        print(f"{len(self.files)} csv files found")
        self.reader = None
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_sentence_len = max_sentence_len

    def batch_generator(self):
        for file_path in self.files:
            with gzip.open(file_path, 'rt') as csvfile:
                data = csv.reader(csvfile, delimiter = ";")
                next(data) # skip header
                for i,row in enumerate(data):
                    e = self.tokenizer(row[1], max_length=self.max_sentence_len, padding='max_length', truncation=True, return_tensors="pt")
                    e['labels'] = self.tokenizer(row[2], max_length=self.max_sentence_len, padding='max_length', truncation=True, return_tensors="pt").input_ids
                    yield i,e
    def __iter__(self):
        return self.batch_generator()

def collate_fn(data):
    # rebuild all samples of a given batch into a dictionary HF way
    batch = {}
    for j,sample in data:
        for k,v in sample.items():
            if batch.get(k) is None: batch[k] = []
            batch[k].append(torch.LongTensor(v))
    batch = {k:torch.vstack(batch[k]) for k in batch.keys()}
    return batch

def train(args, world_size, device):
    print("Starting training job")
    os.makedirs(args.checkpoints_path, exist_ok=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{args.model_name}")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(f"google/{args.model_name}")
    optimizer = AdamW(model.parameters(), lr=args.lr * xm.xrt_world_size())

    train_dataset = QnADataset(args.train, args.max_sentence_len, True, tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)

    best_path = os.path.join(args.checkpoints_path, args.model_name, 'best.pt')
    best_loss = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_time = time.time()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            epoch_loss += outputs.logits.shape[0] * loss.detach().to('cpu')
            num_batches += 1

            # gather gradient updates from all cores and apply them
            xm.optimizer_step(optimizer)
        elapsed = time.time()-epoch_time
        epoch_loss /= num_batches*args.batch_size
        xm.master_print(f"epoch:{epoch}; elapsed_time(sec):{elapsed:0.2f}; loss:{epoch_loss};")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            xm.save({'state_dict': model.state_dict(), 'loss': best_loss}, best_path)

    best_model = torch.load(best_path)
    best_loss = best_model["loss"]
    print(f'Saving best model. Loss: {best_loss}')
    model.load_state_dict(best_model['state_dict'])
    model.to('cpu')
    model.eval()
    model.save_pretrained(args.model_path)
    tokenizer.save_pretrained(args.model_path)

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        inputs = json.loads(request_body)
        return inputs['inputs']

    raise Exception(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps({'answer': prediction})
    raise Exception(f"Unsupported accept: {content_type}")


def model_fn(model_dir):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model,tokenizer

def predict_fn(input_object, model_tokenizer):
    global max_sentence_len,max_new_tokens
    model,tokenizer = model_tokenizer

    input_ids = tokenizer(input_object, max_length=max_sentence_len, padding='max_length', truncation=True, return_tensors="pt").input_ids
    gen_output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in gen_output]

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                prog = 'Train script for Trainium',
                description = 'Hyperparameters for the training process')

    # t5-xxl-ssm" # requires split ~46GB
    parser.add_argument('--num-epochs', type=int, help="Number of epochs", default=2)
    parser.add_argument('--batch-size', type=int, help="Batch size", default=4)
    parser.add_argument('--max-sentence-len', type=int, help="Maximum sentence length", default=128)
    parser.add_argument('--max-new-tokens', type=int, help="Maximum number of generated tokens", default=64)
    parser.add_argument('--model-name', type=str, help="Name of the model", default="t5-large-ssm")
    parser.add_argument('--lr', type=float, help="Learning rate", default=5e-5)

    parser.add_argument('--model-path', type=str, help="Path where we'll save the model", default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--checkpoints-path', type=str, help="Path where we'll save the best model and cache", default='/opt/ml/checkpoints')
    parser.add_argument('--train', type=str, help="Path to train data", default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()
    print(args)
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    import torch_xla.test.test_utils as test_utils
    import torch_xla.distributed.parallel_loader as pl
    
    from torch.optim import AdamW    

    cache_dir = os.path.join(args.checkpoints_path, args.model_name)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['NEURON_CC_FLAGS']=f"--cache_dir={cache_dir} --retry_failed_compilation"
    os.environ['XLA_USE_BF16'] = '1'
    
    device = 'xla'
    # Initialize XLA process group for torchrun
    torch.distributed.init_process_group(device)
    world_size = xm.xrt_world_size()
    
    print(f"Device: {device} World size: {world_size}")
    train(args, world_size, device)

    # define the max_seq len
    with open(__file__, "r") as f:
        data = f.read()
        data = data.replace("\"<MAX_LEN>\"", f"{args.max_sentence_len}")
        data = data.replace("\"<MAX_NEW_TOKENS>\"", f"{args.max_new_tokens}")

    code_path = os.path.join(args.model_path, 'code')
    if not os.path.isdir(code_path): os.makedirs(code_path, exist_ok=True)
    # save a copy of the inference file to the correct dir
    with open(os.path.join(code_path, 'inference.py'), "w") as f:
        f.write(data)  
