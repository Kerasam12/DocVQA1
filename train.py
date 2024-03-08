import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer, T5Model
from DataLoaderVQA import SP_VQADataset
from VQAModel import ModelVT5
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration
from evaluate import accuracy
import wandb

# Initialize WandB
wandb.init(project='SP-DocVQA', name='Basic-DocVQA-OnlyTXT')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


kwargs_dir = 'args.json'
kwargs = {
    "MAX_LEN_STR" :512,
    "TOKENIZER": T5Tokenizer.from_pretrained('t5-small'),
    "MAX_LEN_BBOX":310,
    "MAX_LEN_QUESTION":80,
    "MAX_LEN_ANSWER":10
}

phisical_dev = "PC"
path_strt = "paths.json"

annotations_train_dir = '/data2/users/jsamper/Data/Annotations/train_v1.0_withQT.json'
annotations_val_dir = '/data2/users/jsamper/Data/Annotations/val_v1.0_withQT.json'
ocr_dir = '/data2/users/jsamper/Data/OCR'
images_dir = '/data2/users/jsamper/Data/Images'
# Create an instance of the custom dataset
new_width = 1400
new_height = 1980 
reshape_transform = transforms.Compose([
    transforms.Resize((new_width, new_height)),  # Specify the new dimensions
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    
]) 
#
batch_size = 30#Define the batch size 

#Define the train dataset
train_dataset = SP_VQADataset(annotations_train_dir, ocr_dir, images_dir, transform = reshape_transform,**kwargs)#max_len_answer = MAX_LEN_ANSWER, max_len_question = MAX_LEN_QUESTION, max_len_bbox = MAX_LEN_BBOX, max_len_str=MAX_LEN_STR, tokenizer = TOKENIZER)
# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Define the validation dataset
val_dataset = SP_VQADataset(annotations_val_dir, ocr_dir, images_dir, transform = reshape_transform,**kwargs)#max_len_answer = MAX_LEN_ANSWER, max_len_question = MAX_LEN_QUESTION, max_len_bbox = MAX_LEN_BBOX, max_len_str=MAX_LEN_STR, tokenizer = TOKENIZER)
# Create the DataLoader
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


'''for data in train_loader:#, context_txt, context_bbox, image, answer
    print(data['question'], data['context'],data['context_bbox'], data['image'], data['answer'])'''
model = ModelVT5().to(device)
tokenizer =  T5Tokenizer.from_pretrained('t5-small')
#model_gen = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

learning_rate = 0.00002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 1000

wandb.config.learning_rate = learning_rate
wandb.config.batch_size = batch_size
wandb.config.epochs = epochs
#wandb.watch(model, criterion, log='all')

def validation(model, val_loader):
    for data in val_loader:
        question = data['question'].to(device)
        context = data['context'].to(device)#ocr tokenized ids text
        context_bbox = data['context_bbox'].to(device)
        image = data['image'].to(device)
        answer = data['answer'].to(device)

        output = model.forward(image, context, context_bbox, question)
        out_words = tokenizer.batch_decode(output)
        print(out_words)
        loss = model.model(input_ids=output, labels=answer).loss
        acc = accuracy(output,answer)
        wandb.log({'epoch_val_loss': loss, 'val_accuracy': acc})

        return loss, acc

for step in range(epochs):
    print('epoch:', step)
    tot_loss = 0
    acc_list = []
    model.train()

    if step % 20 == 0:
        print("save model")
        name = "weights" + str(step)
        torch.save(model.state_dict(),"/home/jsamper/DocVQA_Code/model_weights/last_weights.pth" )
    for data in train_loader:
        question = data['question'].to(device)
        context = data['context'].to(device)#ocr tokenized ids text
        context_bbox = data['context_bbox'].to(device)
        image = data['image'].to(device)
        answer = data['answer'].to(device)
        answer = answer.squeeze()

        output = model.forward(image, context, context_bbox, question)
        output = output.to(device)
        
        loss = model.model(input_ids=output, labels=answer).loss
        acc = accuracy(output,answer)
        acc_list.append(acc)
        wandb.log({'individual_train_loss': loss, 'step_accuracy': acc})
        
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(tokenizer.batch_decode(output[0]), tokenizer.batch_decode(answer[0]))
    print(tot_loss)
    avg_acc = sum(acc_list)/len(acc_list)
    wandb.log({'epoch_train_loss': tot_loss, 'avg_epoch_accuracy': avg_acc})
    val_loss, val_acc = validation(model,val_loader)
    
        #out_words = tokenizer.batch_decode(output)
        #print(out_words[0])