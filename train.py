import argparse
import random

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.ImageNet100 import ImageNet100
from templates import imagenet_templates
from utils.non_nv import encode_text_with_learnt_tokens

num_tokens = 77
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt_templates = imagenet_templates 

def optimize_prefix_token(args, object_tokens, model, preprocess):

    asterix_token = clip.tokenize(["*"]).to(device)[0][1]

    trainable_estimated_tokens = torch.nn.Embedding.from_pretrained(object_tokens, freeze=False) #create learnble tokens

    optimizer = optim.SGD(trainable_estimated_tokens.parameters(), lr=args.latent_lr)

    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.latent_ep, eta_min=0.00005, verbose=True)

    torch.cuda.empty_cache()
    trainable_estimated_tokens = optimize_trainable_prefix_token(args, asterix_token, model,
                                                              optimizer, schedular, preprocess, trainable_estimated_tokens)

    res_object_tokens = trainable_estimated_tokens.weight
    return res_object_tokens

def optimize_trainable_prefix_token(args, asterix_token, model,
                                                              optimizer, schedular, preprocess, trainable_estimated_tokens):
    
    training_data = ImageNet100('datasets/ImageNet', split='train', preprocess=preprocess)
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    
    loss_ce = nn.CrossEntropyLoss()
    loss_kd = nn.KLDivLoss(reduction="batchmean")

    #Get token to optimize
    classes = training_data.class_to_idx
    prompt_prefix = [""] * len(classes)
    prompt_original = [""] * len(classes)

    running_loss = 0.0
    for ep in range(args.latent_ep):
        print(f"Start {ep} epoch")
        print(f"Learning rate: {schedular.get_last_lr()[0]}")

        for i, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            prompt = random.choice(prompt_templates)
            for class_i in classes.keys():
                prompt_prefix[classes[class_i]] = prompt.format("* " + class_i)
                prompt_original[classes[class_i]] = prompt.format(class_i)
            token_prefix = clip.tokenize(prompt_prefix).to(device)
            token_original = clip.tokenize(prompt_original).to(device)
            
            original_images, typographic_images, labels = data
            original_images, typographic_images, labels = original_images.to(device), typographic_images.to(device), labels.to(device)

            loss = cross_entropy_kl_loss(original_images, typographic_images, model, token_prefix, token_original, asterix_token, trainable_estimated_tokens, labels, loss_kd, loss_ce, gamma=args.gamma, seta=args.seta, no_reg=args.no_reg)
            print("loss", loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            running_loss = 0.0
            
        schedular.step()

        if (ep + 1) % 3 == 0:
            tokens = trainable_estimated_tokens.weight
            if args.no_reg:
                torch.save(tokens, f"./learned_token/{args.name}_{ep+1}ep_{args.prefix_num}_prefix_{args.latent_lr}_no_reg.pt")
            else:
                torch.save(tokens, f"./learned_token/{args.name}_{ep+1}ep_{args.prefix_num}_prefix_{args.latent_lr}_reg.pt")   

    
    print('Finished Training')

    return trainable_estimated_tokens

def cross_entropy_kl_loss(original_images, typographic_images, model, token_prefix, token_original, asterix_token, trainable_estimated_tokens, labels, loss_kl, loss_ce, gamma, seta=100, temp=1, no_reg=True):
    with torch.no_grad():
        original_image_features = model.encode_image(original_images)
        original_image_features = F.normalize(original_image_features, dim=-1)
        typographic_image_features = model.encode_image(typographic_images)
        typographic_image_features = F.normalize(typographic_image_features, dim=-1)
        
        token_original_features = model.encode_text(token_original)
        token_original_features = F.normalize(token_original_features, dim=-1)

    torch.cuda.empty_cache()
    
    text_prefix_features = model.encode_text_with_learnt_tokens(token_prefix, asterix_token, trainable_estimated_tokens, is_emb = True)
    text_prefix_features = F.normalize(text_prefix_features, dim=-1)

    logit_scale = model.logit_scale.exp()
    logits_prefix = logit_scale * typographic_image_features @ text_prefix_features.t()
    logits_original = logit_scale * original_image_features @ token_original_features.t()
    logits_regularization = logit_scale * original_image_features @ text_prefix_features.t()

    loss1 = loss_ce(logits_prefix, labels)
    loss2 = loss_kl(F.log_softmax(logits_regularization/temp, dim=-1), (logits_original/temp).softmax(dim=-1))
    print('loss1: ', loss1)
    print('loss2: ', loss2)

    if no_reg:
        loss = loss1
    else:
        loss = seta * loss1 + gamma * loss2
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_lr', type=float, default=0.002)
    parser.add_argument('--latent_ep', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--prefix_num', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=3.0)
    parser.add_argument('--seta', type=float, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--no_reg', action='store_true')
    args = parser.parse_args()

    # model, preprocess = clip.load("RN50x4", device=device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Insert CLIP text encoding with learnt token methods
    funcType = type(model.encode_text)
    model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)

    # Initialize prefix token
    prefix_tokens = clip.tokenize(["X"]).to(device)
    prefix_tokens = model.token_embedding(prefix_tokens)[0][1].unsqueeze(0)
    prefix_tokens = torch.empty(args.prefix_num, prefix_tokens.shape[1], dtype=model.dtype, device=device, requires_grad=True)
    nn.init.normal_(prefix_tokens, std=0.02)
    
    prefix_tokens = optimize_prefix_token(args, prefix_tokens, model, preprocess)

    if args.no_reg:
        torch.save(prefix_tokens, f"./learned_token/{args.name}_{args.latent_ep}ep"+f"_{args.prefix_num}_prefix_{args.latent_lr}_no_reg.pt")
    else:
        torch.save(prefix_tokens, f"./learned_token/{args.name}_{args.latent_ep}ep"+f"_{args.prefix_num}_prefix_{args.latent_lr}_reg.pt")   

if __name__ == '__main__':
    main()