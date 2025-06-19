import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load model
model=load_model(r"C:\Machine_Learning\Deep Learning\LSTM_RNN\next_word_lstm.h5")

#tokenizer
with open("tokenizer.pkl","rb") as file:
    tokeniser=pickle.load(file)
    
#prediction function
def predict_next_word(model,tokenizer,data,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([data])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len,padding="pre")
    predicted=model.predict(token_list,verbose=1)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

#streamlit app
st.title("Next word prediction using LSTM RNN")
input_text=st.text_input("Enter Sequence of words:")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]
    next_word=predict_next_word(model,tokeniser,input_text,max_sequence_len)
    st.write(f"Next word is {next_word}")