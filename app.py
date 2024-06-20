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
def plot_loss(losses):
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    st.pyplot(plt)
    st.markdown("""
    **Training Loss Over Epochs**: This graph shows how the model's error decreases over time as it learns from the training data. A lower loss indicates a better performing model.
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
st.markdown("In this step, you will define the parameters for your transformer model. These parameters determine how the model processes and learns from the data.")

# Add explainer for model parameters
d_model_explainer = "A higher number can capture more complex patterns but requires more computational power."
nhead_explainer = "More heads can help the model learn more complex relationships."
num_encoder_layers_explainer = "More layers can improve understanding but require more computation."
num_decoder_layers_explainer = "More layers can help generate more accurate results but require more computation."

# Update slider names to be more descriptive and add explanations
d_model = st.slider("Embedding Size (d_model): Number of features in each word's vector representation.", 128, 512, 256, step=64, help=d_model_explainer)
nhead = st.slider("Number of Attention Heads (nhead): How many parts of the data to focus on at once.", 1, 8, 4, help=nhead_explainer)
num_encoder_layers = st.slider("Number of Encoder Layers: How many layers the model uses to process and understand the input data.", 1, 6, 2, help=num_encoder_layers_explainer)
num_decoder_layers = st.slider("Number of Decoder Layers: How many layers to generate the output.", 1, 6, 2, help=num_decoder_layers_explainer)

vocab_size = 30522  # Using BERT base uncased tokenizer's vocab size

if st.button("Initialize Model"):
    if d_model % nhead == 0:
        model = SimpleTransformerWithDecoder(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers)
        st.session_state.model = model
        st.success("Model initialized!")
    else:
        st.error("Error: Embedding Size (d_model) must be divisible by Number of Attention Heads (nhead)")

# Step 2: Load and preprocess data
st.header("Step 2: Load and Preprocess Data")
st.markdown("In this step, you will load and preprocess the data that the model will be trained on. Preprocessing ensures the data is in the right format for the model.")

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

# Step 3: Train the model
st.header("Step 3: Train the Model")
st.markdown("In this step, you will train the transformer model using the preprocessed data. Training involves adjusting the model's parameters to minimize the error on the training data.")

num_epochs = st.slider("Number of Epochs: Times the training algorithm will pass through the entire training dataset.", 1, 20, 10, help="More epochs can improve model performance but might also lead to overfitting.")
learning_rate = st.slider("Learning Rate: How quickly the model updates its parameters.", 0.0001, 0.01, 0.001, step=0.0001, help="A higher learning rate can speed up training but might overshoot the optimal solution.")

if "model" in st.session_state and "dataloader" in st.session_state:
    model = st.session_state.model
    dataloader = st.session_state.dataloader
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    if st.button("Start Training"):
        losses = []
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
            
            average_loss = epoch_loss / len(dataloader)
            losses.append(average_loss)
            st.write(f"Epoch {epoch+1}, Loss: {average_loss}")

        st.session_state.losses = losses
        st.success("Training completed!")
        plot_loss(losses)

# Step 4: Evaluate the model
st.header("Step 4: Evaluate the Model")
st.markdown("In this step, you can evaluate the trained model by inputting text and observing the generated output. This helps you understand how well the model has learned from the training data.")

input_text = st.text_input("Enter text for evaluation", "I love")

if "model" in st.session_state and st.button("Evaluate"):
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    predicted_text = evaluate(model, tokenizer, input_text)
    st.write(f"Input Text: {input_text}")
    st.write(f"Predicted Continuation: {predicted_text}")

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

# Additional functionalities: by Hugging Face Pipeline
st.header("Additional Functionalities: Hugging Face Pipeline")
st.markdown("These additional functionalities utilize pre-trained models by Hugging Face to perform tasks. These tools can help enhance the performance and capabilities of your model.")

# Masked Language Modeling
st.subheader("Masked Language Modeling")
st.markdown("**Masked Language Modeling**: This technique involves masking a word in a sentence and having the model predict the masked word. It's important for understanding context and improving the model's ability to fill in missing information.")
masked_text = st.text_input("Enter text with [MASK] token", "Hello I'm a [MASK] model.")
if st.button("Fill Mask"):
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    results = unmasker(masked_text)
    st.write(results)

# Feature Extraction
st.subheader("Feature Extraction")
st.markdown("**Feature Extraction**: This process involves converting raw text into numerical features that can be used by machine learning models. It's important because it allows the model to process and understand the input text.")
feature_text = st.text_input("Enter text for feature extraction", "Replace me by any text you'd like.")
if st.button("Extract Features"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    encoded_input = tokenizer(feature_text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    st.write(output)

# Additional information
st.header("Further Learning Resources")
st.markdown("""
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- ["Attention Is All You Need" Paper](https://arxiv.org/abs/1706.03762)
- [Deep Learning Specialization by Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Natural Language Processing with Deep Learning (Stanford)](http://web.stanford.edu/class/cs224n/)
- [Hugging Face's "Transformers" Course](https://huggingface.co/course/chapter1)
""")
