import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader # <-- ADDED: Missing import for DataLoader

# Import your custom modules
from data_loader import FlickrDataset, collate_fn, Vocabulary # <-- ADDED: Import Vocabulary
from model import CNNtoRNN

# <-- ADDED: Helper function to read the text file
def load_captions_from_file(file_path):
    """
    Flickr8k usually comes with a captions.txt file where the first line is a header 
    like 'image,caption', and the rest are 'image_name.jpg,A dog is running...'
    """
    captions_dict = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Skip the first line if it's a header like "image,caption"
        if "image" in lines[0].lower():
            lines = lines[1:]
            
        for line in lines:
            # Split by the first comma only
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                img_id = parts[0]
                caption = parts[1]
                captions_dict.append((img_id, caption))
    return captions_dict

def train():
    # 1. Setup Device (This tells PyTorch to use the Cloud GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Hyperparameters (The knobs you can tune later)
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 5

    # 3. Data Transformations (Resize images to 224x224 for ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 4. Data Loading Pipeline
    # NOTE: You will need to change these paths once you are in Colab!
    captions_dict = load_captions_from_file("captions.txt") 
    
    # Extract just the sentences to build the dictionary
    all_sentences = [caption for img, caption in captions_dict]
    
    # Build the Vocabulary
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_sentences)
    vocab_size = len(vocab)
    
    # Initialize the Dataset and DataLoader
    dataset = FlickrDataset("Images", captions_dict, vocab, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    # 5. Initialize the Model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore the padded zeros!
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 7. The Actual Loop
    model.train() # This tells PyTorch we are training (important for certain layers)

    for epoch in range(num_epochs):
        print(f"--- Starting Epoch {epoch+1}/{num_epochs} ---")
        
        # The dataloader hands us a fresh batch of images and their matching captions
        for idx, (imgs, captions) in enumerate(dataloader):
            
            # Step A: Move the flashcards to the Cloud GPU
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Step B: The Forward Pass (Making a Guess)
            outputs = model(imgs, captions)
            
            # Step C: Calculate the Loss (Grading the Guess)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            # Step D: The Backward Pass (Learning from the Mistake)
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            
            # Step E: Print progress so we know Colab hasn't frozen!
            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
        
        # <-- ADDED: Save the model at the end of every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] complete! Saving checkpoint...")
        torch.save(model.state_dict(), f"baseline_model_epoch_{epoch+1}.pth")
                
    print("Training Complete!")

if __name__ == "__main__":
    train()