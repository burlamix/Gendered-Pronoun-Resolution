#!/bin/bash

source venv/bin/active

pip install --user -r requirements.txt
pip install --user spacy && python -m spacy download en
jupyter lab
