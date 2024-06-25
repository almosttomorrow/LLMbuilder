import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, pipeline
import matplotlib.pyplot as plt
import numpy as np

# Define the transformer model with encoder and decoder
class SimpleTransformerWithDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(SimpleTransformerWithDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        tgt = tgt.permute(1, 0, 2)
        output = self.transformer(src, tgt)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, d_model)
        output = self.fc(output)
        return output

# Define a simple dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return encoded_text['input_ids'].squeeze(), encoded_text['attention_mask'].squeeze()

# Helper function to plot loss
def plot_loss(losses, val_losses=None):
    plt.figure()
    plt.plot(range(len(losses)), losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    st.pyplot(plt)
    st.markdown("""
    **Training and Validation Loss Over Epochs**: This graph shows how the model's error decreases over time as it learns from the training data and is validated on the validation data. A lower loss indicates a better performing model.
    """)

# Helper function for model evaluation
def evaluate(model, tokenizer, text, max_length=10):
    model.eval()
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    input_ids = encoded_text['input_ids']
    tgt_input = input_ids[:, :-1]

    with torch.no_grad():
        output = model(input_ids, tgt_input)

    output_tokens = output.argmax(dim=-1)
    decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return decoded_output

# Streamlit UI
st.title("Build a Transformer Model (LLM) with PyTorch")

# Step 1: Define model parameters
st.header("Step 1: Define Model Parameters")

# Add explainer for model parameters
d_model_explainer = "A higher number can capture more complex patterns but requires more computational power."
nhead_explainer = "More heads can help the model learn more complex relationships."
num_encoder_layers_explainer = "More layers can improve understanding but require more computation."
num_decoder_layers_explainer = "More layers can help generate more accurate results but require more computation."

# Update slider names to be more descriptive and add explanations
d_model = st.slider("Embedding Size (d_model)", 128, 512, 256, step=64, help="Number of features in each word's vector representation. " + d_model_explainer)
nhead = st.slider("Number of Attention Heads (nhead)", 1, 8, 4, help="How many parts of the data to focus on at once. " + nhead_explainer)
num_encoder_layers = st.slider("Number of Encoder Layers", 1, 6, 2, help="How many layers the model uses to process and understand the input data. " + num_encoder_layers_explainer)
num_decoder_layers = st.slider("Number of Decoder Layers", 1, 6, 2, help="How many layers to generate the output. " + num_decoder_layers_explainer)

vocab_size = 30522  # Using BERT base uncased tokenizer's vocab size

if st.button("Initialize Model"):
    if d_model % nhead == 0:
        model = SimpleTransformerWithDecoder(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers)
        st.session_state.model = model
        st.success("Model initialized!")
    else:
        st.error("Error: Embedding Size (d_model) must be divisible by Number of Attention Heads (nhead)")

# Step 2: Load & Preprocess Data
st.header("Step 2: Load & Preprocess Data")

texts = ["I love programming.", "Python is great.", "Transformers are powerful."]
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    st.session_state.tokenizer = tokenizer
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")

if 'tokenizer' in st.session_state:
    tokenizer = st.session_state.tokenizer
    dataset = TextDataset(texts, tokenizer, max_length=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    st.write("**Sample Texts:** These are the example sentences that will be used for training and evaluation.")
    st.write(texts)
    st.write("**Tokenized Sample:** This shows the numerical representation of the sample texts after tokenization, which is necessary for the model to process the data.")
    sample_data = next(iter(dataloader))
    st.write(sample_data)

    if st.button("Preprocess Data"):
        st.session_state.dataloader = dataloader
        st.success("Data preprocessed!")

# Step 3: Train Model
st.header("Step 3: Train Model")

num_epochs = st.slider("Number of Epochs", 1, 500, 10, help="Times the training algorithm will pass through the entire training dataset. More epochs can improve model performance but might also lead to overfitting.")
learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, help="How quickly the model updates its parameters. A higher learning rate can speed up training but might overshoot the optimal solution.")
patience = st.slider("Patience", 1, 10, 3, help="During training, if the validation loss does not improve for a number of consecutive epochs equal to the patience value, the training will stop.")

if "model" in st.session_state and "dataloader" in st.session_state:
    model = st.session_state.model
    dataloader = st.session_state.dataloader
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Split data into training and validation sets
    train_texts = texts[:2]
    val_texts = texts[2:]

    # Create DataLoader for validation set
    val_dataset = TextDataset(val_texts, tokenizer, max_length=10)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    if st.button("Start Training"):
        losses = []
        val_losses = []  # List to store validation losses
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            for src, _ in dataloader:
                tgt_input = src[:, :-1]
                tgt_output = src[:, 1:]

                optimizer.zero_grad()
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Calculate average training loss for the epoch
            average_loss = epoch_loss / len(dataloader)
            losses.append(average_loss)

            # Validation loop
            val_epoch_loss = 0
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                for val_src, _ in val_dataloader:
                    val_tgt_input = val_src[:, :-1]
                    val_tgt_output = val_src[:, 1:]

                    val_output = model(val_src, val_tgt_input)
                    val_loss = criterion(val_output.reshape(-1, vocab_size), val_tgt_output.reshape(-1))
                    val_epoch_loss += val_loss.item()

            # Calculate average validation loss for the epoch
            average_val_loss = val_epoch_loss / len(val_dataloader)
            val_losses.append(average_val_loss)

            st.write(f"Epoch {epoch+1}, Training Loss: {average_loss}, Validation Loss: {average_val_loss}")

            # Early stopping logic
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                st.write(f"Early stopping triggered at epoch {epoch+1}")
                break

            # Set the model back to training mode
            model.train()

        st.session_state.losses = losses
        st.session_state.val_losses = val_losses
        st.success("Training completed!")
        plot_loss(losses, val_losses)

# Step 4: Evaluate Model
st.header("Step 4: Evaluate Model")

input_text = st.text_input("Enter text for evaluation", "I love")

if "model" in st.session_state and st.button("Evaluate"):
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    predicted_text = evaluate(model, tokenizer, input_text)
    st.success(f"**Predicted Next Words:** {predicted_text}")

# Add download button
if "model" in st.session_state:
    torch.save(st.session_state.model.state_dict(), "model.pth")
    with open("model.pth", "rb") as file:
        st.download_button(
            label="Download Model",
            data=file,
            file_name="model.pth",
            mime="application/octet-stream"
        )

# Additional ML Functions
st.header("Additional ML Functions", help="These additional functionalities utilize pre-trained models by Hugging Face to perform tasks. These tools can help enhance the performance and capabilities of your model.")

# State to keep track of visibility
if "show_additional" not in st.session_state:
    st.session_state.show_additional = False

button_label = "Show All" if not st.session_state.show_additional else "Collapse All"
if st.button(button_label):
    st.session_state.show_additional = not st.session_state.show_additional

if st.session_state.show_additional:
    # Masked Language Modeling
    st.subheader("Masked Language Modeling")
    masked_text = st.text_input("Enter text with [MASK] token", "Hello I'm a [MASK] model.", key="masked_text", help="This technique involves masking a word in a sentence and having the model predict the masked word. It's important for understanding context and improving the model's ability to fill in missing information.")
    if st.button("Fill Mask"):
        unmasker = pipeline('fill-mask', model='bert-base-uncased')
        results = unmasker(masked_text)
        st.write(results)

    # Feature Extraction
    st.subheader("Feature Extraction")
    feature_text = st.text_input("Enter text for feature extraction", "Replace me by any text you'd like.", key="feature_text", help="This process involves converting raw text into numerical features that can be used by machine learning models. It's important because it allows the model to process and understand the input text.")
    if st.button("Extract Features"):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        encoded_input = tokenizer(feature_text, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        st.write(output)
