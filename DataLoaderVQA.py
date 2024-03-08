import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import sys
from transformers import T5Tokenizer, T5Model
from torch.nn.functional import normalize


kwargs = {
    "MAX_LEN_STR" :512,
    "TOKENIZER": T5Tokenizer.from_pretrained('t5-small'),
    "MAX_LEN_BBOX":310,
    "MAX_LEN_QUESTION":80,
    "MAX_LEN_ANSWER":20
}

sys.path.append('data2/users/jsamper/Data/OCR')

class SP_VQADataset(Dataset):
    def __init__(self, annotations_dir, ocr_dir, images_dir,transform,  **kwargs): #
        #max_len_bbox,max_len_str, tokenizer, max_len_question,max_len_answer):
        # Initialize the ColorizationDataset class with the specified root directory and transformation
        self.max_len_str = kwargs['MAX_LEN_STR']#max_len_str
        self.max_len_bbox = kwargs['MAX_LEN_BBOX']#max_len_bbox
        #self.eos_char = eos_char
        self.max_len_question = kwargs['MAX_LEN_QUESTION']#max_len_question
        self.max_len_answer = kwargs['MAX_LEN_ANSWER']#max_len_answer
        self.annotations_dir = annotations_dir
    
        self.ocr_dir = ocr_dir
        self.images_dir = images_dir
        self.transform = transform
        self.tokenizer = kwargs['TOKENIZER']#tokenizer 
        self.tokenizer.add_tokens('<no_answ>')
        #self.transform = transform
        # Get a list of image files in the root directory
        self.ocr_files = [f for f in os.listdir(ocr_dir) if os.path.isfile(os.path.join(ocr_dir, f))]
        self.image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    
    def __len__(self):
        # Return the length of the dataset (number of image files)
        
        with open(self.annotations_dir,'r') as annotations:
            
            ann = json.load(annotations)
            
            return len(ann['data'])

    def __getitem__(self, idx):
        # Get the image at the specified index
        with open(self.annotations_dir) as ann:
            annotations = json.load(ann)
            annotations_data = annotations['data'][idx]
            
            image_name = annotations_data['image'][10:]#Pick the image directory and eliminate the directory associated and only keep the image name 
            image_path = os.path.join(self.images_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            ocr_name = image_name[:-3] + 'json'#Erase the 'png' part and replace it with 'json'
            ocr_path = os.path.join(self.ocr_dir, ocr_name)
            ocr_route = open(ocr_path)
            ocr = json.load(ocr_route)
             
            
            #ocr_list = ocr['recognitionResults'][0]['lines']
            #data = ocr['recognitionResults'][0]['lines']
            question, questionId = self.get_questions(annotations_data)
            context,context_bbox,context_emb = self.process_ocr(ocr)
            answer_encoded, answer, start_answ_idx, end_answ_idx = self.get_start_end_answer_idx(context, annotations_data)
          
        
        return {'question':question,
                'context': context_emb,
                'context_bbox': context_bbox,
                'image':image,
                'answer':answer_encoded}
        #(question, context_bbox, context_txt, image, answer)#, context_bbox, image, answer
    
    def process_ocr(self, ocr):
        context = [txt['text'] for txt in ocr['recognitionResults'][0]['lines']]#get all the text in the image by sentences recognized by the OCR
        context_bbox = []
        context_txt = []
        #padding_bbox = torch.tensor([[0,0], [0,0], [0,0], [0,0]])
        padding_bbox = torch.tensor([0,0, 0,0, 0,0, 0,0])
        #max_bb = 290
        
        
        
        data = ocr['recognitionResults'][0]['lines']
        #for data in ocr:
        for d in data:
            
            context_bbox.append(d['boundingBox'])#get all the bounding boxes of the text in the image by words 
            context_txt.append(d['text'])#get all the text in the image by words 
            
        pad_len_bbox = self.max_len_bbox - len(context_bbox) 
        #pad_len_txt = max(0,self.max_len_str - len(context_txt))
        #pad_list = [self.eos_char]
        context_txt = ' '.join(context_txt)
        caption_encoded = self.tokenizer.encode(context_txt, max_length=self.max_len_str, padding='max_length', return_attention_mask=True, return_token_type_ids=False, truncation=True,return_tensors = 'pt')
        caption_encoded =caption_encoded.squeeze(0)
        
        #print(len(context_bbox), pad_len_bbox)
        expanded_bbox = padding_bbox.unsqueeze(0).repeat(pad_len_bbox, 1)
        
        if len(context_bbox) != 0:
            context_bbox = torch.tensor(context_bbox).reshape((len(context_bbox),-1))#4,2))
            min_values, _ = torch.min(context_bbox, dim=1, keepdim=True)
            max_values, _ = torch.max(context_bbox, dim=1, keepdim=True)
            context_bbox =(1980 * (context_bbox - min_values) / (max_values - min_values)).round().to(torch.int32)

            #print(expanded_bbox.shape)
            #print(context_bbox.shape)
            context_bbox = torch.cat([context_bbox, expanded_bbox], dim=0)# Concatenate the padding along the first dimension   
        else:
            context_bbox = expanded_bbox
        #print(context_txt)
        return context,context_bbox,caption_encoded

    def get_questions(self, annotations_data):
        question = annotations_data['question']
        question_encoded = self.tokenizer.encode(question, max_length=self.max_len_question, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True,return_tensors = 'pt')
        questionId = annotations_data['questionId']
        return question_encoded, questionId
    
    
    def get_start_end_answer_idx(self, context, annotations_data):
        answers = annotations_data['answers']
        
        answers_encoded = []
        for answ in answers:
            
            answers_encoded.append(self.tokenizer.encode(answ, max_length=self.max_len_answer,padding='max_length', return_attention_mask=True, return_token_type_ids=False, truncation=True,return_tensors = 'pt'))
            
        
        context_joined = "".join(context)

        answer_positions = []
        for answer in answers:
            start_idx = context_joined.find(answer)

            if start_idx != -1:
                end_idx = start_idx + len(answer)
                answer_positions.append([start_idx, end_idx])

        if len(answer_positions) > 0:
            start_idx, end_idx = random.choice(answer_positions)  # If both answers are in the context. Choose one randomly.
            answer = context_joined[start_idx: end_idx]
        else:
            start_idx, end_idx = 0, 0  # If the indices are out of the sequence length they are ignored. Therefore, we set them as a very big number.

        return answers_encoded[0], answers,start_idx, end_idx
        