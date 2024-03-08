import torch 
import torch.nn as nn
from utils import VisualEmbeddings, SpatialEmbeddings
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelVT5(nn.Module):
    def __init__(self):
        super(ModelVT5, self).__init__()
        self.visual_embedding = VisualEmbeddings()
        self.spatial_embedding = SpatialEmbeddings()
        self.semantic_embedding = T5EncoderModel.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.decoder_start_token_id = self.tokenizer.pad_token_id
        

    def concat_embeddings(self,input_embeds,visual_embed):
        #semantic_embed = semantic_embed['last_hidden_state']
        #input_embeds = torch.cat((semantic_embed, spatial_embed), dim=1)
        input_embeds = torch.cat([input_embeds, visual_embed], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
        
        
        return input_embeds
    
    def encoder(self,image, ocr, ocr_bounding_box, question):
        #input_ids = self.tokenizer(ocr) 
        visual_embed, visual_emb_mask = self.visual_embedding(image)
        #for ocr_samp, ocr_bbx in zip(ocr, ocr_bounding_box): 
            
        '''spatial_embed = self.spatial_embedding(ocr_bounding_box)

        semantic_embed = self.semantic_embedding(input_ids = ocr)

        
        
        input_embeds = torch.add(semantic_embed, spatial_embed)
        input_embeds = torch.cat([input_embeds, visual_embed], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)
        encoder_embed = []
        encoder_embed = self.concat_embeddings(semantic_embed, spatial_embed, visual_embed)'''
        
        input_embeds = torch.tensor([]).to(device)
        for ocr_samp, ocr_bbx in zip(ocr, ocr_bounding_box): 
            #print(ocr_bbx.shape)
            #print(ocr_samp.shape)
            ocr_bbx = ocr_bbx.unsqueeze(0)
            ocr_samp = ocr_samp.unsqueeze(0)
            spatial_embed = self.spatial_embedding(ocr_bbx)

            semantic_embed = self.semantic_embedding(input_ids = ocr_samp)
            semantic_embed = semantic_embed['last_hidden_state']
            embeds = torch.cat((semantic_embed, spatial_embed), dim=1)
            
            input_embeds = torch.concatenate((input_embeds, embeds))
            
            
            '''input_embeds = torch.add(semantic_embed, spatial_embed)
            input_embeds = torch.cat([input_embeds, visual_embed], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
            tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)
            encoder_embed = []'''
        
        question = question.squeeze()
        question_embed = self.semantic_embedding(input_ids = question)
           
        encoder_embed = self.concat_embeddings(input_embeds, visual_embed)
        encoder_embed = torch.cat([question_embed['last_hidden_state'], visual_embed], dim=1)

        return encoder_embed
    
    def decoder(self, encoder_embed):
        outputs = self.model.generate(inputs_embeds= encoder_embed,decoder_start_token_id=self.decoder_start_token_id, max_new_tokens = 512)
        return outputs
        
    def forward(self,image, ocr, ocr_bounding_box, question):
        encoder_embed = self.encoder(image, ocr, ocr_bounding_box, question)
        output = self.decoder(encoder_embed)
        return output