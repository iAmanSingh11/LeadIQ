import streamlit as st
from app import predict, df, bundle, lr, rf, feature_names

st.title("Test")
st.write("Import working!")
st.write(df.shape)