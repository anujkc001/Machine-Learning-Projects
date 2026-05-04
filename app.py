import streamlit as st
import json
import numpy as np
import pickle
from tensorflow import keras

st.title("Jupyter-Hosted Chatbot")

# Your existing chat logic here...
# (Paste the full Streamlit code I gave you in the previous response here)


# Page Configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("My AI Assistant")

# --- LOAD ASSETS (Cached so they only load once) ---
@st.cache_resource
def load_chat_assets():
    with open("intents.json") as file:
        data = json.load(file)
    model = keras.models.load_model('chat_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    return data, model, tokenizer, lbl_encoder

data, model, tokenizer, lbl_encoder = load_chat_assets()

# --- CHAT HISTORY INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Predict response
    max_len = 20
    seq = tokenizer.texts_to_sequences([prompt])
    padded = keras.preprocessing.sequence.pad_sequences(seq, truncating='post', maxlen=max_len)
    result = model.predict(padded, verbose=0)

    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    # Get random response from JSON
    response = "I'm sorry, I don't understand."
    for i in data['intents']:
        if i['tag'] == tag:
            response = np.random.choice(i['responses'])

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
