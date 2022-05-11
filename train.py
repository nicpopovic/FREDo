# %%
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.utils.logging import enable_propagation
import string
import random
from torch.utils.data import DataLoader
from src.data import parse_episodes, collate_fn_train, parse_episodes_from_index
import os
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import wandb
import argparse
from src.models.util import set_seed
from src.util import get_f1, get_f1_macro
from tqdm import tqdm

# %%
 
random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)


parser = argparse.ArgumentParser()
parser.add_argument("--seed_model", type=int, default=123, help="random seed for model")
parser.add_argument("--use_markers", type=bool, default=True, help="use entity marker")
parser.add_argument("--seed_data", type=int, default=123, help="random seed for data")
parser.add_argument("--num_epochs", type=int, default=25, help="number of epochs to train")
parser.add_argument("--support_docs_train", type=int, default=1, help="number of support documents during training")
parser.add_argument("--support_docs_eval", type=int, default=1, help="number of support documents during eval")
parser.add_argument("--query_docs_train", type=int, default=3, help="number of query documents (train)")
parser.add_argument("--query_docs_eval", type=int, default=3, help="number of query documents (eval)")
parser.add_argument("--samples_per_ep", type=int, default=2000, help="number of samples per epoch")
parser.add_argument("--samples_data_train", type=int, default=50000, help="number of training episodes to generate")
parser.add_argument("--samples_data_dev", type=int, default=500, help="number of dev episodes to generate")
parser.add_argument("--samples_data_test", type=int, default=10000, help="number of test episodes to generate")
parser.add_argument("--balancing_train", type=str, default="single", help="balancing (hard, soft, single)")
parser.add_argument("--balancing_eval", type=str, default="single", help="balancing (hard, soft, single)")
parser.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size")
parser.add_argument("--train_batch_size", type=int, default=2, help="training batch size")
parser.add_argument("--warmup_epochs", type=int, default=1, help="warmup epochs")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate")
parser.add_argument("--loss", type=str, default="atloss", help="loss function")
parser.add_argument("--ensure_positive", type=bool, default=True, help="ensure positive example query")
parser.add_argument("--load_checkpoint", type=str, default=None, help="path to checkpoint")

parser.add_argument("--project", type=str, default="FREDo", help="project name for wandb")
parser.add_argument("--model", type=str, default="dlmnav", help="model")

args = parser.parse_args()
wandb.init(project=args.project)
wandb.config.update(args)
wandb.config.identifier = random_string

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# %%

train_only = False
markers = args.use_markers

samples_per_epoch = args.samples_per_ep
n_epochs = args.num_epochs
train_batch_size = args.train_batch_size
warmup_epochs = args.warmup_epochs
learning_rate = args.learning_rate

if args.num_epochs != 0:
    training_episodes = parse_episodes("data/train.json", tokenizer, K=args.support_docs_train, n_queries=args.query_docs_train, n_samples=args.samples_data_train, markers=args.use_markers, balancing=args.balancing_train, seed=args.seed_data, ensure_positive=args.ensure_positive, cache=None)
    dev_episodes = parse_episodes("data/dev.json", tokenizer, K=args.support_docs_eval, n_queries=args.query_docs_eval, n_samples=args.samples_data_dev, markers=args.use_markers, balancing=args.balancing_eval, seed=args.seed_data, ensure_positive=args.ensure_positive, cache=None)
indomain_test_episodes = parse_episodes_from_index("data/test_docred.json", f"data/test_in_domain_{args.support_docs_eval}_doc_indices.json", tokenizer, markers=args.use_markers, cache=None)
scierc_test_episodes = parse_episodes_from_index("data/test_scierc.json", f"data/test_cross_domain_{args.support_docs_eval}_doc_indices.json", tokenizer, markers=args.use_markers, cache=None)
# %%
if args.model != "knnproto":
    torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
g = torch.Generator()
g.manual_seed(args.seed_data)
set_seed(args.seed_model)

if args.num_epochs != 0:

    train_loader = DataLoader(
        training_episodes, 
        batch_size=train_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_train, num_workers=4, drop_last=True, generator=g)

else:
    train_loader = []
if not train_only and args.num_epochs != 0:

    dev_loader = DataLoader(
        dev_episodes, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_train, num_workers=4, drop_last=False)

indomain_test_loader = DataLoader(
        indomain_test_episodes, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_train, num_workers=4, drop_last=False)

scierc_test_loader = DataLoader(
        scierc_test_episodes, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_train, num_workers=4, drop_last=False)

# %%


# -------------- model loading --------------

# start by loading language model

lm_config = AutoConfig.from_pretrained(
    "bert-base-cased",
    num_labels=10,
)
lm_model = AutoModel.from_pretrained(
    "bert-base-cased",
    from_tf=False,
    config=lm_config,
)

# %%
if args.model == "dlmnav":
    from src.models.dlmnav import Encoder
elif args.model == "dlmnav+sie":
    from src.models.dlmnav_sie import Encoder
elif args.model == "dlmnav+sie+sbn":
    from src.models.dlmnav_sbn import Encoder

encoder = Encoder(
    config=lm_config,
    model=lm_model, 
    cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
    sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
    markers=markers
    )

encoder.to('cuda')


if args.load_checkpoint is not None:
    print(f'loading model from {args.load_checkpoint}')
    encoder.load_state_dict(torch.load(f"{args.load_checkpoint}"))



pretrained = encoder.model.parameters()
pretrained_names = [f'model.{k}' for (k, v) in encoder.model.named_parameters()]
new_params= [v for k, v in encoder.named_parameters() if k not in pretrained_names]
optimizer = AdamW(encoder.parameters(), lr=learning_rate, eps=1e-6)
scaler = GradScaler()
num_samples = len(train_loader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_epochs * samples_per_epoch/train_batch_size), int(samples_per_epoch/train_batch_size*n_epochs))

step_global = -1

train_iter = iter(train_loader)

best_f1 = 0.0
for i in tqdm(range(n_epochs)):
    true_positives, false_positives, false_negatives = {},{},{}

    encoder.train()
    loss_agg = 0
    count = 0
    with tqdm(range(int(samples_per_epoch/train_batch_size))) as pbar:
        for _ in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            step_global += 1
            exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
            with autocast():
                output, loss = encoder(exemplar_tokens.to('cuda'), exemplar_mask.to('cuda'), exemplar_positions, exemplar_labels, query_tokens.to('cuda'), query_mask.to('cuda'), query_positions, query_labels, label_types)
                #output, loss = encoder(exemplar_tokens.to('cuda'), exemplar_mask.to('cuda'), exemplar_positions, exemplar_labels, exemplar_tokens.to('cuda'), exemplar_mask.to('cuda'), exemplar_positions, exemplar_labels, label_types)
            #loss = encoder(exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels)
            
            for pred, lbls in zip(output, query_labels):
                for preds, lbs in zip(pred, lbls):
                    for inf in preds:
                        if inf[2] not in true_positives.keys():
                            true_positives[inf[2]] = 0
                            false_positives[inf[2]] = 0
                            false_negatives[inf[2]] = 0

                        if inf in lbs:
                            true_positives[inf[2]] += 1
                        else:
                            false_positives[inf[2]] += 1

                    for label in lbs:
                        if label[2] not in true_positives.keys():
                            true_positives[label[2]] = 0
                            false_positives[label[2]] = 0
                            false_negatives[label[2]] = 0

                        if label not in preds:
                            false_negatives[label[2]] += 1

            #p,r,f = get_f1(true_positives, false_positives, false_negatives)
            #print(true_positives, false_positives, false_negatives)
            count += 1
            loss_agg += loss.item()
            pbar.set_postfix({"Loss":f"{loss_agg/count:.2f}"})

            wandb.log({"loss": loss.item()}, step=step_global)
            wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]}, step=step_global)

            # backpropagate & reset
            scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # clip gradients
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)

            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()
            lr_scheduler.step()
            encoder.zero_grad()
            del loss, output

    p,r,f = get_f1(true_positives, false_positives, false_negatives)
    p_train, r_train, f1_train = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)
    wandb.log({"precision_train": p_train}, step=step_global)
    wandb.log({"recall_train": r_train}, step=step_global)
    wandb.log({"f1_macro_train": f1_train}, step=step_global)
    wandb.log({"f1_micro_train": f}, step=step_global)
    
    if not train_only:

        true_positives, false_positives, false_negatives = {},{},{}


        encoder.eval()
        with tqdm(dev_loader) as pbar:
            for batch in pbar:
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
                output = encoder(exemplar_tokens.to('cuda'), exemplar_mask.to('cuda'), exemplar_positions, exemplar_labels, query_tokens.to('cuda'), query_mask.to('cuda'), query_positions, None, label_types)
                #loss = encoder(exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels)
                
                for pred, lbls in zip(output, query_labels):
                    for preds, lbs in zip(pred, lbls):
                        for inf in preds:
                            if inf[2] not in true_positives.keys():
                                true_positives[inf[2]] = 0
                                false_positives[inf[2]] = 0
                                false_negatives[inf[2]] = 0

                            if inf in lbs:
                                true_positives[inf[2]] += 1
                            else:
                                false_positives[inf[2]] += 1

                        for label in lbs:
                            if label[2] not in true_positives.keys():
                                true_positives[label[2]] = 0
                                false_positives[label[2]] = 0
                                false_negatives[label[2]] = 0

                            if label not in preds:
                                false_negatives[label[2]] += 1

        p,r,f = get_f1(true_positives, false_positives, false_negatives)
        p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)

        if f1_dev >= best_f1:
            wandb.log({"best_precision_dev": p_dev}, step=step_global)
            wandb.log({"best_recall_dev": r_dev}, step=step_global)
            wandb.log({"best_f1_macro_dev": f1_dev}, step=step_global)
            wandb.log({"Best_f1_micro_dev": f}, step=step_global)
            best_f1 = f1_dev
            torch.save(encoder.state_dict(), f"checkpoints/{args.project}_{random_string}.pt")

        wandb.log({"precision_dev": p_dev}, step=step_global)
        wandb.log({"recall_dev": r_dev}, step=step_global)
        wandb.log({"f1_macro_dev": f1_dev}, step=step_global)
        wandb.log({"f1_micro_dev": f}, step=step_global)
# %%

print("---- INDOMAIN TEST EVAL -----")
if n_epochs > 0:
    encoder.load_state_dict(torch.load(f"checkpoints/{args.project}_{random_string}.pt"))
else:
    step_global = 0

true_positives, false_positives, false_negatives = {},{},{}


encoder.eval()

if args.model == "dlmnav+sie+sbn":
    encoder.dev = False

with tqdm(indomain_test_loader) as pbar:
    for batch in pbar:
        exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
        output = encoder(exemplar_tokens.to('cuda'), exemplar_mask.to('cuda'), exemplar_positions, exemplar_labels, query_tokens.to('cuda'), query_mask.to('cuda'), query_positions, None, label_types)
        #loss = encoder(exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels)
        
        for pred, lbls in zip(output, query_labels):
            for preds, lbs in zip(pred, lbls):
                for inf in preds:
                    if inf[2] not in true_positives.keys():
                        true_positives[inf[2]] = 0
                        false_positives[inf[2]] = 0
                        false_negatives[inf[2]] = 0

                    if inf in lbs:
                        true_positives[inf[2]] += 1
                    else:
                        false_positives[inf[2]] += 1

                for label in lbs:
                    if label[2] not in true_positives.keys():
                        true_positives[label[2]] = 0
                        false_positives[label[2]] = 0
                        false_negatives[label[2]] = 0

                    if label not in preds:
                        false_negatives[label[2]] += 1

p,r,f = get_f1(true_positives, false_positives, false_negatives)
p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)

wandb.log({"precision_test_indomain": p_dev}, step=step_global)
wandb.log({"recall_test_indomain": r_dev}, step=step_global)
wandb.log({"f1_macro_test_indomain": f1_dev}, step=step_global)
wandb.log({"f1_micro_test_indomain": f}, step=step_global)

print("---- SCIERC TEST EVAL -----")

true_positives, false_positives, false_negatives = {},{},{}

with tqdm(scierc_test_loader) as pbar:
    for batch in pbar:
        exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
        output = encoder(exemplar_tokens.to('cuda'), exemplar_mask.to('cuda'), exemplar_positions, exemplar_labels, query_tokens.to('cuda'), query_mask.to('cuda'), query_positions, None, label_types)
        #loss = encoder(exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels)
        
        for pred, lbls in zip(output, query_labels):
            for preds, lbs in zip(pred, lbls):
                for inf in preds:
                    if inf[2] not in true_positives.keys():
                        true_positives[inf[2]] = 0
                        false_positives[inf[2]] = 0
                        false_negatives[inf[2]] = 0

                    if inf in lbs:
                        true_positives[inf[2]] += 1
                    else:
                        false_positives[inf[2]] += 1

                for label in lbs:
                    if label[2] not in true_positives.keys():
                        true_positives[label[2]] = 0
                        false_positives[label[2]] = 0
                        false_negatives[label[2]] = 0

                    if label not in preds:
                        false_negatives[label[2]] += 1

p,r,f = get_f1(true_positives, false_positives, false_negatives)
p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)

wandb.log({"precision_test_scierc": p_dev}, step=step_global)
wandb.log({"recall_test_scierc": r_dev}, step=step_global)
wandb.log({"f1_macro_test_scierc": f1_dev}, step=step_global)
wandb.log({"f1_micro_test_scierc": f}, step=step_global)
