import numpy as np
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertConfig 
from transformers import AlbertConfig ,AlbertTokenizer,RobertaTokenizer,RobertaConfig
from pytorch_transformers import WarmupLinearSchedule
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from BertModules import BertClassifier
from Constants import *
from DataModules import SequenceDataset
from Utils import seed_everything
import seaborn as sns
from matplotlib import pyplot as plt


seed_everything()

# Load BERT default config object and make necessary changes as per requirement
# config = RobertaConfig(MODEL_PATH)


if MODEL_TYPE == 'bert':
    config = BertConfig(hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    num_labels=2)
if MODEL_TYPE == 'albert':
    config = AlbertConfig(hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    num_labels=2)
if MODEL_TYPE == 'roberta':
    config = RobertaConfig(hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    num_labels=2)


# Create our custom BERTClassifier model object
model = BertClassifier(config)
model.to(DEVICE)
# model.bert.from_pretrained(MODEL_PATH, config=config,ignore_mismatched_sizes=True)
# Initialize BERT tokenizer
if MODEL_TYPE == 'albert':
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
if MODEL_TYPE == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
if MODEL_TYPE == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Load Train dataset and split it into Train and Validation dataset
train_dataset = SequenceDataset(TRAIN_FILE_PATH, tokenizer)

validation_split = 0.2
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
shuffle_dataset = True

if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

print ('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))

# Loss Function
criterion = nn.CrossEntropyLoss()

# Adam Optimizer with very small learning rate given to BERT
optimizer = torch.optim.Adam([
    {'params': model.bert.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 3e-4}
])

# Learning rate scheduler
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS,
                                 t_total=len(train_loader) // GRADIENT_ACCUMULATION_STEPS * NUM_EPOCHS)

model.zero_grad()
epoch_iterator = trange(int(NUM_EPOCHS), desc="Epoch")
training_acc_list, validation_acc_list = [], []

# model.load_state_dict(torch.load('./modelset/model2.pth')) 
for epoch in epoch_iterator:
    epoch_loss = 0.0
    train_correct_total = 0

    # Training Loop
    train_iterator = tqdm(train_loader, desc="Train Iteration")
    for step, batch in enumerate(train_iterator):
        model.train(True)
        # Here each element of batch list refers to one of [input_ids, segment_ids, attention_mask, labels]
        inputs = {
            'input_ids': batch[0].to(DEVICE),
            'token_type_ids': batch[1].to(DEVICE),
            'attention_mask': batch[2].to(DEVICE)
        }

        labels = batch[3].to(DEVICE)
        logits = model(**inputs)

        loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        epoch_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scheduler.step()
            optimizer.step()
            model.zero_grad()

        _, predicted = torch.max(logits.data, 1)
        # print(predicted)
        correct_reviews_in_batch = (predicted == labels).sum().item()
        train_correct_total += correct_reviews_in_batch


    print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))
    pred_list = []
    label_list = []
    # Validation Loop
    with torch.no_grad():
        val_correct_total = 0
        model.train(False)
        val_iterator = tqdm(val_loader, desc="Validation Iteration")
        for step, batch in enumerate(val_iterator):
            inputs = {
                'input_ids': batch[0].to(DEVICE),
                'token_type_ids': batch[1].to(DEVICE),
                'attention_mask': batch[2].to(DEVICE)
            }

            labels = batch[3].to(DEVICE)
            logits = model(**inputs)

            _, predicted = torch.max(logits.data, 1)
            pred_list.extend(predicted.cpu())
            label_list.extend(labels.cpu())
            correct_reviews_in_batch = (predicted == labels).sum().item()
            val_correct_total += correct_reviews_in_batch

        training_acc_list.append(train_correct_total * 100 / len(train_indices))
        validation_acc_list.append(val_correct_total * 100 / len(val_indices))
        print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
            train_correct_total * 100 / len(train_indices), val_correct_total * 100 / len(val_indices)))


# torch.save(model.state_dict(), './modelset/model3.pth')

acc = metrics.accuracy_score(label_list, pred_list)
print("accuracy:",acc)
r = metrics.recall_score(label_list, pred_list)
print("recall:",r)
pre_score = metrics.precision_score(label_list, pred_list)  
print("precision",pre_score)
f1=metrics.f1_score(label_list, pred_list)
print("F1-score",f1)


cm = confusion_matrix(label_list, pred_list)

sns.set(font_scale=2.0)
ax = sns.heatmap(cm, annot=True, fmt='g'
#, cmap='Blues')
)
ax.title.set_text("Confusion Matrix")
ax.set_xlabel("pred")
ax.set_ylabel("true")
plt.savefig("./cm/{}.jpg".format(MODEL_TYPE))



# text = 'I am a big fan of cricket'
# text = '[CLS] ' + text + ' [SEP]'
#
# encoded_text = tokenizer.encode(text) + [0] * 120
# tokens_tensor = torch.tensor([encoded_text])
# labels = torch.tensor([1])
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([
#                 {'params': model.bert.parameters(), 'lr' : 1e-5},
#                 {'params': model.classifier.parameters(), 'lr': 1e-3}
#             ])


# logits = model(tokens_tensor, labels=labels)
# loss = criterion(logits, labels)
# print(loss)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

