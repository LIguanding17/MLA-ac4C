import streamlit as st
from PIL import Image
from functions import *
from model.model import Model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import random

st.title("""********* MLA-ac4C *********""")
st.subheader(
    """MLA-ac4C is a high-performance N4-Acetylcytidine site prediction model based on a combination of Multi-layer Attention, Bi-GRU, and DenseNet architectures.""")

image = Image.open('model.png')
st.image(image)


def predict():
    # Initialize the session state for the sequence if not already set
    if 'sequence' not in st.session_state:
        st.session_state.sequence = ""

    # Button to automatically populate the text area with the sample sequence
    if st.button('Sample Sequence'):
        df = pd.read_csv('dataset/train_data.csv')
        sample = random.choice(df['Sequence'])
        st.session_state.sequence = sample  # Assign sample sequence to session state

    # Text area for sequence input (with the option to be filled by the sample sequence)
    sequence = st.text_area("Sequence Input", value=st.session_state.sequence, height=200)
    st.session_state.sequence1 = sequence  # Update session state with user input

    # Submit button logic
    if st.button("Submit"):
        # Fill in 0 labels to fit model input
        df = pd.DataFrame({
            "Sequence": [sequence],
            "Label": [0]
        })
        df.to_csv("./dataset/input.csv", index=False)

        file_path = 'dataset/input.csv'
        features, labels = process_csv_and_encode(file_path)

        dataset = TensorDataset(features, labels)
        val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device={device}')
        model = Model().to(device)

        model.load_state_dict(torch.load(f'./model/model.pth', map_location=device))
        model.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, unit='batch'):
                batch_data, batch_labels = batch
                batch_data, batch_labels = batch_data.to(device).float(), batch_labels.to(device).float()
                outputs = model(batch_data)

        if outputs > 0.5:
            st.info("N4-Acetylcytidine Site")
        else:
            st.info("Non-N4-Acetylcytidine Site")


predict()
