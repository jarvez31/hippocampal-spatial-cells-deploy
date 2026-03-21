import streamlit as st
import requests
import data

st.title("Spatial Cell Representation")

env = st.selectbox("Environment", ["aligned_lattice", "tilted_lattice", "helix", "pegboard"])

if st.button("Run"):
    trajectory = data.load_trajectory(env, "/Users/b/Downloads/Deploy_model/")
    response = requests.post("http://localhost:8000/results", json={"trajectory": trajectory.tolist()})
    
    if response.status_code == 200:
        st.session_state["results"] = response.json()
    else:
        st.error("Failed to get results from the server.")

if "results" in st.session_state:
    st.write(st.session_state["results"])