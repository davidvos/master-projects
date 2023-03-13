import torch

def expand_to_batchsize_for_layer(tup, batch_size, layer_id):
    return tup[layer_id].expand(-1, batch_size, -1, -1, -1)

def generate_data(prefix_model, dataloader, tokenizer, device, epoch, lr, preseqlen, hidden_dim):
    print(f'Generate eval file epoch for {epoch}')
    with open(f'../results/multilingual-prefix-tuning/epoch{epoch}_lr{lr}_preseqlen{preseqlen}_hiddendim{hidden_dim}.txt', 'w') as f:
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                prefix_model.eval()

                samples = batch[0].squeeze().to(device)

                if len(list(samples.shape)) == 1:
                    samples = torch.unsqueeze(samples, 0)

                # Get Past-Key-Values
                past_key_values = prefix_model(batch_size=samples.shape[0])
                prefix_model.past_key_values = past_key_values

                outputs = prefix_model.model.generate(samples)
                f.write(f'{tokenizer.decode(outputs[0], skip_special_tokens=True)}')
                f.write('\n')