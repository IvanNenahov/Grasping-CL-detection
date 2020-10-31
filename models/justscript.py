from models.mobilenetLR import *

model = MobileNetWLR(pretrained=True)

dataset = CORE50(root='../core50_128x128', scenario="nicv2_391")
device = torch.device("cuda:0")
model.to(device)

for idx, batch in enumerate(dataset):

    print(f'train on {idx} batch')
    model.train_on_data(batch)

test_x, test_y = dataset.get_test_set()

ave_loss, acc, accs = get_accuracy(
        model, torch.nn.CrossEntropyLoss(), 128, test_x, test_y, preproc=preprocess_imgs
    )
