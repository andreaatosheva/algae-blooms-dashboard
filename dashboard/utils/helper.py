import streamlit as st
import psutil

def show_memory_usage():
    mem = psutil.virtual_memory()
    with st.sidebar:
        st.caption(f"🧠 Memory: {mem.percent}% ({mem.used / 1024**2:.0f} MB)")