#!/bin/bash
apt-get update -y
apt-get install -y libsm6 libxext6 libxrender-dev
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
