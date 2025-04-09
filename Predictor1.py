import streamlit as st
from PIL import Image
from functions import *
from model import Model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

st.title(
    """
           ********* MLA-ac4C *********
    """
)
# st.subheader("""5-Methylcytosine (m5c) is a modified cytosine base which is formed as the result of addition of methyl group added at position 5 of carbon. This modification is one of the most common PTM that used to occur in almost all types of RNA. The conventional laboratory methods do not provide quick reliable identification of m5c sites. However, the sequence data readiness has made it feasible to develop computationally intelligent models that optimize the identification process for accuracy and robustness. The present research focused on the development of in-silico methods built using ensemble models. The encoded data was then fed into ensemble models, which included bagging and boosting ensemble model as well. After that, the models were subjected to a rigorous evaluation process that included both independent set testing and 10-fold cross validation. The results revealed that Bagging ensemble model, outperformed revealing 100% accuracy while comparing with existing m5c predictors.
# """)
image = Image.open('model.png')
st.image(image)


def predict():
    # st.subheader("Input Sequence of any length")

    # Initialize the session state for the sequence if not already set
    if 'sequence' not in st.session_state:
        st.session_state.sequence = ""

    # Sample RNA sequence
    sample = "UGGUGUACUCCAGCAACUUCCAGAACGUGAAGCAGCUGUACGCGCUGGUGUGCGAAACGCAGCGCUACUCCGCCGUGCUGGAUGCUGUGAUCGCCAGCGCCGGCCUCCUCCGUGCGGAGAAGAAGCUGCGGCCGCACCUGGCCAAGGUGCUAGUGUAUGAGUUGUUGUUGGGAAAGGGCUUUCGAGGGGGUGGGGGCCGAU"

    # Button to automatically populate the text area with the sample sequence
    if st.button('Sample Sequence'):
        st.session_state.sequence = sample  # Assign sample sequence to session state

    # Text area for sequence input (with the option to be filled by the sample sequence)
    sequence = st.text_area("Sequence Input", value=st.session_state.sequence, height=200)
    st.session_state.sequence1 = sequence  # Update session state with user input

    # Display the current value of the sequence
    # st.write(st.session_state.sequence1)

    # Submit button logic
    if st.button("Submit"):
        # st.write(st.session_state.sequence1)  # Display the sequence when submitted

        # abc will be assigned the current value of sequence1 (whether manually entered or auto-filled)
        # abc = str(sequence1)
        # st.write(sequence1)
        # st.write(sequence1)
        # st.write(f"Submitted Sequence: {abc}")
        # 构造 DataFrame
        df = pd.DataFrame({
            "Sequence": [sequence],
            "Label": [0]
        })

        # 保存为 CSV 文件
        df.to_csv("./data/input.csv", index=False)

        file_path = './data/input.csv'  # 输入的CSV文件
        features, labels = process_csv_and_encode(file_path)

        dataset = TensorDataset(features, labels)
        val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

        # 实例化模型、定义损失函数和优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device={device}')
        model = Model().to(device)

        model.load_state_dict(torch.load(f'./model/model.pth'))
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
