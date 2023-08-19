import argparse

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import dataset
from utils.non_nv import encode_text_with_learnt_tokens

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_on_dataset(args, model, preprocess, prefix_tokens):

    test_data = dataset(args, preprocess)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(test_data.templates[0].format(c)) for c in test_data.classes]).to(device)
    text_inputs = text_inputs.cuda()

    text_prefix_token = torch.cat([clip.tokenize(test_data.templates[0].format(f'* {c}')) for c in test_data.classes]).to(device)
    asterix_token = clip.tokenize(["*"]).to(device)[0][1]

    model.eval()

    acc, acc_baseline, total = 0, 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
    
        text_prefix_feature = model.encode_text_with_learnt_tokens(text_prefix_token, asterix_token, prefix_tokens.unsqueeze(0), is_emb=False)
        text_prefix_feature = F.normalize(text_prefix_feature, dim=-1)
        
        for _, (original_image, typographic_image, target) in enumerate(test_loader):
            if args.evaluate_on_TA:
                image = typographic_image.to(device)
            else:
                image = original_image.to(device)
            target = target.to(device)

            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity = 100. * (img_features @ text_prefix_feature.T)
            similarity_baseline = 100. * (img_features @ text_feature.T)
        
            probs = F.softmax(similarity, dim=-1).max(-1)[1]
            
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]

            acc += probs.eq(target).sum().item()
            acc_baseline += probs_baseline.eq(target).sum().item()

            total += target.size(0)
    return acc / total, acc_baseline / total
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens_path', type=str, default='./learned_token/experiment4_100ep_1_prefix_0.002_SGD_reg.pt')
    parser.add_argument('-t', '--evaluate_on_TA', action='store_true')
    args = parser.parse_args()

    model, preprocess = clip.load("ViT-B/32", device=device)

    #Inser CLIP text encoding with learnt token methods
    funcType = type(model.encode_text)
    model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)

    #load already inferred tokens
    prefix_tokens = torch.load(args.tokens_path)
    print("args.tokens_path", args.tokens_path)

    dataset_name = ['imagenet', 'caltech', 'pets', 'cars', 'flowers', 'food', 'aircraft', 'dtd', 'sun', 'eurosat']
    # dataset_name = ['disentanglement', 'paint', 'rta-100']

    acc_CLIP, acc_ours = [], []
    for dataset_i in dataset_name:
        args.dataset = dataset_i
        accuracy, accuracy_baseline = eval_on_dataset(args, model, preprocess, prefix_tokens)
        acc_CLIP.append(round(accuracy_baseline*100, 2))
        acc_ours.append(round(accuracy*100, 2))

    acc_CLIP.append(round((sum(acc_CLIP) / len(acc_CLIP)), 2))
    acc_ours.append(round((sum(acc_ours) / len(acc_ours)), 2))

    
    print('& '.join(list(map(str, acc_CLIP))))
    print('& '.join(list(map(str, acc_ours))))

if __name__ == '__main__':
    main()