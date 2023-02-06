from sgnlp.models.sentic_gcn import(
    SenticGCNBertConfig,
    SenticGCNBertModel,
    SenticGCNBertEmbeddingConfig,
    SenticGCNBertEmbeddingModel,
    SenticGCNBertTokenizer,
    SenticGCNBertPreprocessor,
    SenticGCNBertPostprocessor
)
from string import punctuation
import re
import torch
import torch.nn.functional as F

# configuration - initialising
tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")
config = SenticGCNBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json")
model = SenticGCNBertModel.from_pretrained("https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", config=config)
embed_config = SenticGCNBertEmbeddingConfig.from_pretrained("bert-base-uncased")
embed_model = SenticGCNBertEmbeddingModel.from_pretrained("bert-base-uncased", config=embed_config)
preprocessor = SenticGCNBertPreprocessor(tokenizer=tokenizer, embedding_model=embed_model, senticnet="https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle", device="cpu")
postprocessor = SenticGCNBertPostprocessor()

def sentiment_analysis(sentences:list, process_softmax:bool=False):
    # processing inputs
    inputs = []
    inputs_aspect_count = []
    for sentence in sentences:
        # format sentence
        sentence = sentence.lower()
        re.sub(r"\s+", " ", sentence)
        for p in punctuation:
            sentence = sentence.replace(f' {p}', p).replace(f'{p} ', p).replace(p, f' {p} ')
        
        # use each word as 1 aspect (except punctuation)
        sentence_split = sentence.split(' ')
        aspects = [word for word in sentence_split if word not in punctuation]

        # add to inputs
        inputs.append( {'sentence': sentence, 'aspects': list(dict.fromkeys(aspects))} )
        inputs_aspect_count.append(len(aspects))
    print(inputs)
    _, processed_indices = preprocessor(inputs)

    # processing outputs
    # each 'aspect' returns a list of 3 values, corresponding to the probabilities that the aspect is negative, neutral or positive respectively
    raw_outputs = model(processed_indices)
    probabilities = raw_outputs.logits
    if process_softmax:
        probabilities = F.softmax(raw_outputs.logits, dim=-1)
    processed_outputs = list(torch.split(probabilities, inputs_aspect_count, dim=0))

    # note: in one sentence, outputs are reurned in order of aspect, by appearance
    # e.g. aspects are [a, b, c] and sentence is "b a b c", output is in the order "a b b c"
    # output is a list of tensors, 1 tensor = 1 sentence, 1 list in tensor = 1 occurence of aspect
    return processed_outputs

data = sentiment_analysis(["</z>", "You look great today!", "</z>You look great today!"], process_softmax=True)
print(data)