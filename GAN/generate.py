import torch
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Generator(torch.nn.Module):
    def __init__(self,feature_size):
        super().__init__()
        self.tokens=AutoTokenizer.from_pretrained('facebook/opt-350m', padding_side='left', use_fast=True)
        config=AutoConfig.from_pretrained('facebook/opt-350m')
        self.text_model=OPTForCausalLM(config).from_pretrained('facebook/opt-350m')
        self.embeds=self.text_model.get_input_embeddings()
        hidden_size=self.embeds.embedding_dim
        self.features_to_embed=torch.nn.Sequential(torch.nn.Linear(feature_size+1,(feature_size+hidden_size+1)//2),torch.nn.GELU(),torch.nn.Linear((feature_size+hidden_size+1)//2,hidden_size))
    
    def forward(self, image_features, attitude, starting_text = None, max_length = 10, temperature = 1.0, return_probs = False, echo_input_text = False):
        if starting_text==None:
            starting_text = ['']*len(attitude)

        if not (len(image_features)==len(attitude)==len(starting_text)):
            raise ValueError('Batch size must be equal across all arguments')

        tokens=self.tokens(starting_text,padding=True,return_tensors='pt')
        all_new_toks,mask=tokens['input_ids'],torch.cat([torch.ones(len(attitude),1),tokens['attention_mask']],dim=1)
        out_probs=torch.tensor([[]]*len(attitude))

        for i in range(max_length):
            in_embeds=self._generate_embeddings_and_masks(all_new_toks,image_features,attitude,input_is_tokens=True)
            new_tok_logits=self.text_model(inputs_embeds=in_embeds,attention_mask=mask).logits[:,-1,:]
            tok_probs=F.gumbel_softmax(new_tok_logits,tau=temperature,dim=-1)
            new_toks=torch.distributions.categorical.Categorical(probs=tok_probs).sample().unsqueeze(1)
            if return_probs:
                out_probs=torch.cat([out_probs, torch.gather(tok_probs, 1, new_toks)],dim=1)
            all_new_toks=torch.cat([all_new_toks,new_toks],dim=1)
            if all((torch.count_nonzero(toks == self.tokens.eos_token_id) > 1) for toks in all_new_toks):
                break
            mask=torch.cat([mask,torch.ones(len(mask),1)],dim=1)
        all_new_toks=torch.cat([all_new_toks, torch.full((len(all_new_toks),1),self.tokens.eos_token_id,dtype=torch.int)],dim=1)
        if not echo_input_text:
            all_new_toks=torch.cat([torch.full((len(all_new_toks),1),self.tokens.eos_token_id),all_new_toks[:,(-i-2):]], dim=1)
        prob_indices=[(toks == self.tokens.eos_token_id).nonzero(as_tuple=True)[0][1]-len(toks) for toks in all_new_toks]
        all_new_toks=[tok[1:index] for tok,index in zip(all_new_toks,prob_indices)]
        if not return_probs:
            return [self.tokens.decode(toks,skip_special_tokens=True) for toks in all_new_toks]
        out_probs=[(prob[:index+1] if index+1 < 0 else prob) for prob,index in zip(out_probs,prob_indices)]
        return all_new_toks, out_probs
        
            
    def _generate_embeddings_and_masks(self, previous_text, image_features, attitude, input_is_tokens=False):
        features=torch.cat([image_features,attitude],dim=1)
        if input_is_tokens:
            toks={'input_ids':previous_text}
        else:
            toks=self.tokens(previous_text,padding=True,return_tensors='pt')
        previous_embed=self.embeds(toks['input_ids'])
        starting_embed=self.features_to_embed(features).unsqueeze(1)
        #this is for if you dont want the image or attitude to affect text generation
        # if input_is_tokens:
        #     return previous_embed
        # return previous_embed, toks['attention_mask']
        if input_is_tokens:
            return torch.cat([starting_embed,previous_embed],dim=1)
        return torch.cat([starting_embed,previous_embed],dim=1), torch.cat([torch.ones(len(attitude),1),toks['attention_mask']],dim=1)
