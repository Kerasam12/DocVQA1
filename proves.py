import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer, T5Model
from DataLoaderVQA import SP_VQADataset
from VQAModel import ModelVT5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


kwargs_dir = 'args.json'
kwargs = {
    "MAX_LEN_STR" :512,
    "TOKENIZER": T5Tokenizer.from_pretrained('t5-small'),
    "MAX_LEN_BBOX":290,
    "MAX_LEN_QUESTION":80,
    "MAX_LEN_ANSWER":50
}
annotations_dir = '/home/jsamper/Desktop/DocVQA/Data/Annotations/train_v1.0_withQT.json'
ocr_dir = '/home/jsamper/Desktop/DocVQA/Data/OCR'
images_dir = '/home/jsamper/Desktop/DocVQA/Data/Images'
# Create an instance of the custom dataset
new_width = 1400
new_height = 1980
reshape_transform = transforms.Compose([
    transforms.Resize((new_width, new_height)),  # Specify the new dimensions
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    
]) 


MAX_LEN_QUESTION = 80
MAX_LEN_ANSWER = 50

train_dataset = SP_VQADataset(annotations_dir, ocr_dir, images_dir, transform = reshape_transform, **kwargs)#max_len_answer = MAX_LEN_ANSWER, max_len_question = MAX_LEN_QUESTION, max_len_bbox = MAX_LEN_BBOX, max_len_str=MAX_LEN_STR, tokenizer = TOKENIZER)

batch_size = 10
# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



'''for data in train_loader:#, context_txt, context_bbox, image, answer
    print(data['question'], data['context'],data['context_bbox'], data['image'], data['answer'])'''
model = ModelVT5().to(device)
tokenizer =  T5Tokenizer.from_pretrained('t5-small')


for data in train_loader:
    question = data['question'].to(device)
    context = data['context'].to(device)#ocr tokenized ids text
    context_bbox = data['context_bbox'].to(device)
    image = data['image'].to(device)
    answer = data['answer'].to(device)

    output = model.forward(image, context, context_bbox)
    out_words = tokenizer.batch_decode(output)
    print(out_words[0])