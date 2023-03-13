from sys import prefix
from transformers import MT5ForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import logging
import torch
import gc

from datasets import WebNLG
from models import PrefixTuning
from utils import generate_data
      
gc.collect()
torch.cuda.empty_cache()

def main(n_epochs=50, lr=5e-5, accum=32, preseqlen=5, hidden_dim=512, batch_size=2):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = WebNLG(language='both', split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = WebNLG(language='both', split='dev')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

    # Load Pre-Trained Tokenizer, LM
    pretrained = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    pretrained = pretrained.to(device)
    pretrained.resize_token_embeddings(len(train_dataset.tokenizer))
    
    prefix_model = PrefixTuning(model=pretrained, preseqlen=preseqlen, hidden_dim=hidden_dim)
    prefix_model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, prefix_model.parameters()), lr=lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.03 * n_epochs * len(train_dataloader) / (accum * batch_size)),
        num_training_steps=int(n_epochs * len(train_dataloader) / (accum * batch_size))
    )

    writer = SummaryWriter()

    step_global = 0

    for epoch in range(1, n_epochs + 1):

        print('Running epoch: {}'.format(epoch))

        loss_train = 0

        for step, batch in enumerate(train_dataloader):
             
            print(f'{step}/{len(train_dataloader)}')

            prefix_model.train()

            samples = batch[0].squeeze().to(device)
            summaries = batch[1].squeeze().to(device)

            optimizer.zero_grad()

            # Get Past-Key-Values
            past_key_values = prefix_model(batch_size=samples.shape[0])
            prefix_model.past_key_values = past_key_values

            # Forward: Base (Pre-Trained) LM
            outputs = prefix_model.model(input_ids=samples, labels=summaries)

            loss = outputs[0] / accum
            loss.backward()
            loss_train += loss.item()
            
            if (step + 1) % accum == 0:

                step_global+=1
                
                # TensorBoard
                writer.add_scalar(
                    f'loss_train/prefix-tuning-preseqlen{preseqlen}_hidden{hidden_dim}_batch{batch_size * accum}_lr{lr}_epoch{n_epochs}',
                    loss_train,
                    step_global
                )

                # Set Loss to 0
                loss_train = 0

                optimizer.step()
                optimizer.zero_grad()

        generate_data(prefix_model, test_dataloader, test_dataset.tokenizer, device, epoch, lr, preseqlen, hidden_dim)

if __name__ == "__main__":
    main()