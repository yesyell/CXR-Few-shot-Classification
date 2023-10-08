import torch
import clip
import os
from PIL import Image
import pickle
from clip.model import ModifiedResNet
from torch.utils.data import DataLoader
from utils import CustomImageDataset
from torchvision import transforms
from utils import *
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

batch_size = 1

# GPU Setting
GPU_NUM = 2 # Specify GPU to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# load pickle files
with open('../pkl/FINDINGS.pkl', 'rb') as f:
    FINDINGS = pickle.load(f)


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

pretrained_dict = {k[7:]: v for k, v in state_dict.items() if k[7:] in model.state_dict()}

# load pretrained model
model.load_state_dict(pretrained_dict)


transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

# load custom dataset for pretraining
img_dir = '/nas/user/jaeryung/mimic-cxr/'
train_csv = './train_server_10shot.csv'
val_csv = './val_server_10shot.csv'
training_set = CustomImageDataset(train_csv, img_dir, transform)
validation_set = CustomImageDataset(val_csv, img_dir, transform, sample_limit=20)
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

optimzer = optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmupRestarts(optimizer, 1)

def train_one_epoch(epoch_index, tb_writer, text_features):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        image, label = data # label.dtype: torch.int64
        y = one_hot_encode(label)
        image, y = image.to(device), y.to(device)
        #print("DONE input setting")
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #print("DONE optimizer setting")

        # Make predictions for this batch - get s
        img_features = model(image)
        img_features = img_features.to(torch.float16)
        img_features = normalize_features(img_features)
        # img_features: torch.float32
        
        # cosine similarity as logits
        #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #logit_scale = logit_scale.exp()
        logit_scale = 1

        s = logit_scale * img_features @ text_features#.t()
        # outputs = model(inputs)
        #print("DONE cosine similarity compution")

        # Compute the loss and its gradients
        loss = LSES(s, y)
        #print("DONE loss compution")
        #print("Loss: ", loss)
        #loss = loss.unsqueeze(-1)
        loss.mean().backward()
        #print("DONE loss backpropagation")

        # Adjust learning weights
        optimizer.step()
        #print("DONE optimizer.step()")

        # Gather data and report
        running_loss += loss.item()
        if i % batch_size == batch_size - 1:
            last_loss = running_loss / batch_size # loss per batch
            #print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss



# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000. # ?
text_features = get_text_emb(FINDINGS)

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer, text_features)
    scheduler.step()

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()


    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vimage, vlabel = vdata
            vy = one_hot_encode(vlabel)
            vimage, vy = vimage.to(device), vy.to(device)

            # Make predictions for this batch - get s
            img_features = model(vimage)
            img_features = img_features.to(torch.float16)
            img_features = normalize_features(img_features)

            text_features = get_text_emb(FINDINGS)

            logit_scale = 1

            s = logit_scale * img_features @ text_features#.t()

            vloss = LSES(s, vy)
            running_vloss += vloss.mean()

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    epoch_number += 1


torch.save(model.state_dict(), './checkpoints/classifier_model_10shot.pt')
