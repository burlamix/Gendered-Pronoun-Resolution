#!/bin/bash

source venv/bin/active

pip install -r requirements.txt
pip install spacy && python -m spacy download en
jupiter lab
