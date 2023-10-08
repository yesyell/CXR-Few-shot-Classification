import torch
import clip
from PIL import Image
import pickle
from clip.model import ModifiedResNet
from torchvision import transforms
import os
from utils import *


one_shot = True

# GPU Setting
GPU_NUM = 3 # Specify GPU to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

pred = dict()
label = dict()

# load pickle files
with open('../pkl/FINDINGS.pkl', 'rb') as f:
    FINDINGS = pickle.load(f)

with open('../pkl/test_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

if one_shot:
    with open('../pkl/one-shot-class.pkl', 'rb') as f:
        one_shot_class = pickle.load(f)
else: # ten_shot
    with open('../pkl/ten-shot-class.pkl', 'rb') as f:
        ten_shot_class = pickle.load(f)
# load test files
IMG_PATHES = (os.listdir('../test_img/'))

# load state_dict files for models
model_path = '../checkpoints/CLIP_model_10000_3000.pt'
with open(model_path, 'rb') as opened_file:
    state_dict = torch.load(opened_file, map_location="cpu")

state_dict = state_dict['model_state_dict']
counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
vision_layers = tuple(counts)
vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
vision_heads = vision_width * 32 // 64
embed_dim = state_dict["text_projection"].shape[1]

model_clip = clip.model.build_model(state_dict).to(device)
model = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads) # args: layers, output_dim, heads
model = model.to(device)

# define transformation as same as training
transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

# test model
for img_id in IMG_PATHES:
    img_path = '../test_img/' + img_id
    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # Make predictions for this batch - get s
        img_features = model(image)
        img_features = img_features.to(torch.float16)
        img_features = normalize_features(img_features)

        text_features = get_text_emb(FINDINGS)

        logit_scale = 1

        s = logit_scale * img_features @ text_features#.t()

    #print(s)
    pred[img_id] = torch.topk(s, 10)

for img in IMG_PATHES:
    print("PREDICTION")
    ppth = []
    _, idxs = pred[img]

    for i in list(idxs[0]):
        ppth.append(str(FINDINGS[int(i)].split()[0])+ " "+ str(FINDINGS[int(i)].split()[1]))
    print(set(sorted(ppth)))
    print("LABEL")
    lpth = []
    for i in label[img]:
        if int(i) in one_shot_class:
            lpth.append(str(FINDINGS[int(i)].split()[0]) + " " + str(FINDINGS[int(i)].split()[1]))
    print(set(sorted(lpth)))
    print()
