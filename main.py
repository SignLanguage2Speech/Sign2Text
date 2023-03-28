import os
import torch
from Sign2Text import Sign2Text
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

### params ###
save_path = os.path.join('/work3/s204138/bach-models', 'PHOENIX_trained_models')
visual_model_checkpoint = os.path.join(save_path, '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc')
visual_model_vocab_size = 1085
n_visual_features = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mbart_path = '/work3/s200925/mBART/checkpoints/checkpoint-42576'

print("MODEL")
model = Sign2Text(
    visual_model_vocab_size = visual_model_vocab_size, 
    visual_model_checkpoint = visual_model_checkpoint, 
    n_visual_features = n_visual_features, 
    mbart_path = mbart_path,
    device = device)

# print(model.predict(torch.rand(2, 3, 250, 244, 244).to(device)))
# print(model(torch.rand(1, 3, 25, 244, 244).to(device)))

def train(model, epochs):

    x = torch.rand(2, 3, 120, 244, 244).to(device)
    y = ["wetter ist nicht sehr schön heute abend","morgenstag sind es regenlich und schwer"]
    tokenized_y = model.tokenizer(text_target = y, return_tensors = "pt", padding = 'max_length', max_length = 30, add_special_tokens = False).get("input_ids").to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor = 1.0, end_factor = 0.1, total_iters = 1000, verbose = False)

    print(model.predict(x))

    for epoch in range(epochs):
        y_pred, probs = model(x)
        print(y_pred[:,0])
        print(model.tokenizer.decode(torch.argmax(y_pred[0],dim=1)))
        print(model.tokenizer.decode(torch.argmax(y_pred[1],dim=1)))
        y_pred_permute = y_pred.permute(0,2,1)
        loss = loss_fn(y_pred_permute, tokenized_y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(model.predict(x))
    x = torch.rand(1, 3, 120, 244, 244).to(device)
    print(model.predict(x))

train(model, 0)