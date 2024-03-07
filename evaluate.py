import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer, T5Model
from DataLoaderVQA import SP_VQADataset
from VQAModel import ModelVT5
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration

'''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


kwargs_dir = 'args.json'
kwargs = {
    "MAX_LEN_STR" :512,
    "TOKENIZER": T5Tokenizer.from_pretrained('t5-small'),
    "MAX_LEN_BBOX":310,
    "MAX_LEN_QUESTION":80,
    "MAX_LEN_ANSWER":50
}
annotations_dir = '/home/jsamper/Desktop/DocVQA/Data/Annotations/test_v1.0.json'
ocr_dir = '/home/jsamper/Desktop/DocVQA/Data/OCR'
images_dir = '/home/jsamper/Desktop/DocVQA/Data/Images'
# Create an instance of the custom dataset
new_width = 1400
new_height = 1980 
reshape_transform = transforms.Compose([
    transforms.Resize((new_width, new_height)),  # Specify the new dimensions
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    
]) 
#
validation_dataset = SP_VQADataset(annotations_dir, ocr_dir, images_dir, transform = reshape_transform,**kwargs)#max_len_answer = MAX_LEN_ANSWER, max_len_question = MAX_LEN_QUESTION, max_len_bbox = MAX_LEN_BBOX, max_len_str=MAX_LEN_STR, tokenizer = TOKENIZER)

batch_size = 16
# Create the DataLoader
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)'''


def accuracy(preds, ground_truth):
    
    #for pred, gt in zip(preds, ground_truth):
    length = min(ground_truth.shape[1], preds.shape[1])
    bs, _ = ground_truth.shape
    print(ground_truth[::, :length].shape, preds[::, :length].shape)
    correct_predictions = torch.sum((ground_truth[::, :length] == preds[::, :length])& (ground_truth[::, :length] != 0))
    total_samples = len(ground_truth)
    print(total_samples)
    accuracy = correct_predictions.item() / (length * bs)
    return accuracy


def ansl(self):
    pass

