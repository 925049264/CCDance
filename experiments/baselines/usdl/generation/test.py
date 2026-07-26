"""Test script for USDL generation model."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch, numpy as np
from shared.data_loader import load_all_pose_sequences, load_teacher_comments, create_data_splits
from shared.metrics import compute_generation_metrics
from model import USDLGenerator, CommentTokenizer, build_vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    pose_seqs, meta, valid_idx = load_all_pose_sequences()
    comments = load_teacher_comments()
    train_idx, _, test_idx = create_data_splits(valid_idx, meta, seed=args.seed)

    train_comments = [comments[i] for i in train_idx if comments.get(i)]
    tokenizer = build_vocab(train_comments)
    model = USDLGenerator(vocab_size=len(tokenizer.word2idx)).to(device)

    ckpt_path = args.checkpoint or f'{Path(__file__).parent}/seed_{args.seed}/model.pt'
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Model loaded from {ckpt_path}")

    refs, cands = [], []
    for i in test_idx:
        p = torch.FloatTensor(pose_seqs[i]).unsqueeze(0).to(device)
        T = p.shape[1]
        if T > 300:
            idx = np.linspace(0, T-1, 300, dtype=int); p = p[:, idx]
        elif T < 300:
            p = torch.cat([p, torch.zeros(1, 300-T, p.shape[2]).to(device)], dim=1)
        with torch.no_grad():
            out = model.generate(p, tokenizer)
        refs.append(comments.get(i, ""))
        cands.append(out)
    metrics = compute_generation_metrics(refs, cands)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
