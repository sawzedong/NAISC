import torch
from peekingduck_process.preprocess_image import Preprocessor
from GAN.discern import Discerner
from GAN.generate import Generator
from dataloader import ImageOnlyDataLoader, ImageTextDataLoader

if torch.cuda.is_available():  
    device = "cuda:0"
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0, device=device)
else: 
    device = "cpu" 

print(device)

#Generator initialised with feature_size set to 10 as that is the size for mtcnn
G=Generator(10).to(device)
D=Discerner(device=device).to(device)

#hyperparemeters, tune these
epochs=10000
max_sequence_length = 20
monte_carlo_iterations=10
sampling_temperature = 100.0
monte_carlo_sampling_temperature = 100.0
decay_rate=1.1
G_lr=0.01
D_lr=0.01

# model checkpointing parameters
epoch_checkpoint=10 # saves model every epoch_checkpoint epochs
data_content_folder = "" # leave blank for local system, use /content/NAISC/ for google colab
model_folder = r'/content/drive/MyDrive/NAISC_MODEL_CHECKPOINTS/' #PLEASE change accordingly to where you want to save your model

# training datasets
ImageOnlyDataset=ImageOnlyDataLoader(data_content_folder+"annDataset") 
ImageTextDataset=ImageTextDataLoader(data_content_folder+"annDataset/Annotations.json") 

#try different optimizers, should be a drag and drop replacement
G_optim=torch.optim.Adam(G.parameters(),lr=G_lr,amsgrad=True)
D_optim=torch.optim.Adam(D.parameters(),lr=D_lr,amsgrad=True)

initial_epoch = 0
load_model_from_checkpoint = False
if load_model_from_checkpoint:

    G_checkpoint = torch.load((model_folder + "generator.pt"))
    G.load_state_dict(G_checkpoint['model_state_dict'])
    G_optim.load_state_dict(G_checkpoint['optimizer_state_dict'])
    ImageOnlyDataset=ImageOnlyDataLoader(data_content_folder+"annDataset", data_queue=G_checkpoint['loader_queue'], random_generator=G_checkpoint['loader_rng'])
    del G_checkpoint

    D_checkpoint = torch.load((model_folder + "discerner.pt"))
    D.load_state_dict(D_checkpoint['model_state_dict'])
    D_optim.load_state_dict(D_checkpoint['optimizer_state_dict'])
    initial_epoch = D_checkpoint['epoch']+1
    ImageTextDataset=ImageTextDataLoader(data_content_folder+"annDataset/Annotations.json", data_queue=D_checkpoint['loader_queue'], random_generator=D_checkpoint['loader_rng']) 
    del D_checkpoint


preprocess=Preprocessor()
torch.random.manual_seed(0)
skipped=0
#NOT batched because i dont care

for epoch in range(initial_epoch, epochs):
    try:
        print(f"EPOCH {epoch+1}")
        features=[]
        while features==[]:
            image=next(ImageOnlyDataset)
            image, features=preprocess(image)
            if features:
                features=torch.tensor(features[0],dtype=torch.float).unsqueeze(0)
        attitudes=(2*torch.rand(1,1)-1)
        G_optim.zero_grad()
        toks, probs = G.forward(features, attitudes,max_length=max_sequence_length,temperature=sampling_temperature+1,return_probs=True)
        rewards=torch.tensor([])
        with torch.no_grad():
            realness=[]
            for remaining_length in range(1,len(toks[0])):
                new_text=G.forward(features.repeat(monte_carlo_iterations,1),attitudes.repeat(monte_carlo_iterations,1),G.tokens.batch_decode([t[:remaining_length] for t in toks],skip_special_tokens=True)*monte_carlo_iterations,temperature=monte_carlo_sampling_temperature+1,return_probs=True,max_length=max_sequence_length-remaining_length,echo_input_text=True)[0]
                realness.append(torch.mean(D.forward(image*monte_carlo_iterations,G.tokens.batch_decode(new_text,skip_special_tokens=True),attitudes.repeat(monte_carlo_iterations,1))[:][0]).unsqueeze(0))
            final_text=G.tokens.batch_decode(toks,skip_special_tokens=True)
            rewards=torch.cat(realness+[D.forward(image,final_text,attitudes)[0]],dim=0)
        
        print(final_text[0])
        G_loss=-torch.sum(rewards*probs[0])
        print("Generator loss:", G_loss)
        G_loss.backward()
        G_optim.step()

        print('Cooling')
        sampling_temperature /= decay_rate
        monte_carlo_sampling_temperature /= decay_rate
        print('Temperatures:',sampling_temperature, monte_carlo_sampling_temperature)

        D_optim.zero_grad()
        D_loss=D.forward(image,final_text,attitudes)[0][0]
        print("Discerner loss on fake data:", D_loss)
        D_loss.backward()
        D_optim.step()

        D_optim.zero_grad()
        d_image,d_text,d_attitude=next(ImageTextDataset)
        D_loss=-D.forward([d_image],[d_text],[d_attitude])[0][0]
        print("Discerner loss on real data:", D_loss)
        D_loss.backward()
        D_optim.step()

    except (ValueError,TypeError):
        print('SOMETHING HAS GONE WRONG PLEASE FIX THIS SOON')
        skipped+=1
        print(f'{skipped} epochs have been skipped')
    
    if (epoch+1) % epoch_checkpoint == 0 or epoch == epochs:
        torch.save({
            'epoch': epoch,
            'model_state_dict': G.state_dict(),
            'optimizer_state_dict': G_optim.state_dict(),
            'loss': G_loss,
            'loader_queue': ImageOnlyDataset.data_queue,
            'loader_rng': ImageOnlyDataset.rng
        }, (model_folder + "generator.pt"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': D.state_dict(),
            'optimizer_state_dict': D_optim.state_dict(),
            'loss': D_loss,
            'loader_queue': ImageTextDataset.data_queue,
            'loader_rng': ImageTextDataset.rng
        }, (model_folder + "discerner.pt"))
print(f'{skipped} epochs have been skipped')
