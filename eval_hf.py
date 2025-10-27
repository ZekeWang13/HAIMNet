from net.HAIMNet import HAIMNet
import os, json, argparse
import torch, torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import safetensors.torch as sf

torch.backends.cudnn.benchmark = True

def load_weights(model: torch.nn.Module, path_or_repo: str):
    if os.path.isfile(path_or_repo):
        if path_or_repo.endswith((".safetensors", ".sft")):
            state = sf.load_file(path_or_repo)
        else:
            state = torch.load(path_or_repo, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    repo_id = path_or_repo
    try:
        model_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors", repo_type="model")
        state = sf.load_file(model_file)
        model.load_state_dict(state, strict=False)
        return model
    except Exception:
        pass
    model_file = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", repo_type="model")
    state = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model

def save_enhanced(model, img_path, out_dir="./output_hf",
                  gamma=1.0, alpha_s=1.0, alpha_i=1.0,
                  divisor=32, gated=False, gated2=False):
    model.eval()
    if hasattr(model, "trans"):
        if gated:   model.trans.gated = True
        if gated2:  model.trans.gated2 = True
        if hasattr(model.trans, "alpha_s"): model.trans.alpha_s = alpha_s
        if hasattr(model.trans, "alpha"):   model.trans.alpha   = alpha_i

    to_tensor = transforms.ToTensor()
    img = Image.open(img_path).convert("RGB")
    x = to_tensor(img)  # [C,H,W], float32, [0,1]

    c, h, w = x.shape
    H = (h + divisor - 1) // divisor * divisor
    W = (w + divisor - 1) // divisor * divisor
    padh, padw = H - h, W - w
    x = F.pad(x.unsqueeze(0), (0, padw, 0, padh), mode="reflect")

    with torch.no_grad():
        y = model((x.cuda()) ** gamma)
        y = torch.clamp(y, 0, 1)[:, :, :h, :w]

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(img_path)
    out_path = os.path.join(out_dir, name)
    transforms.ToPILImage()(y.squeeze(0).cpu()).save(out_path)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/dataC/iedA/wzq/HAIMNet/train_record/final_best_trans/Blur/DarkIR.pth")
    parser.add_argument("--input_img", type=str, default="/dataC/iedA/wzq/HAIMNet/datasets/LOL_blur/test/low_blur/0127/0066.png")
    parser.add_argument("--alpha_s", type=float, default=1.0)
    parser.add_argument("--alpha_i", type=float, default=1.0)
    parser.add_argument("--gamma",   type=float, default=1.8)
    parser.add_argument("--lol_v1", action="store_true")
    parser.add_argument("--v2",     action="store_true")
    args = parser.parse_args()

    model = HAIMNet().cuda()
    model = load_weights(model, args.path)

    gated  = args.lol_v1 or ("lolv1" in args.path.lower())
    gated2 = args.v2     or ("lolv2" in args.path.lower()) or ("syn" in args.path.lower()) or ("real" in args.path.lower())

    out = save_enhanced(
        model, args.input_img,
        gamma=args.gamma, alpha_s=args.alpha_s, alpha_i=args.alpha_i,
        divisor=32, gated=gated, gated2=gated2
    )
    print("Saved:", out)
