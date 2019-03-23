# Gendered-Pronoun-Resolution

## setup (development mode)

Requires python version 3.6+

	python -m pip install --user -e .

## usage

### computing embeddings 

	hltproject compute-embeddings [-h] [-t TARGET] input

	positional arguments:
	  input                 input filename

	optional arguments:
	  -h, --help            show this help message and exit
	  -t TARGET, --target TARGET
							target directory (default: embeddings)

### baselines

	hltproject baseline-cosine [-h] [-t TARGET] test

	positional arguments:
	  test                  test filename

	optional arguments:
	  -h, --help            show this help message and exit
	  -t TARGET, --target TARGET
							target directory (default: predictions)

### computing loss for a prediction
	hltproject loss [-h] model input

	positional arguments:
	  model       model predictions
	  input       input dataset

	optional arguments:
	  -h, --help  show this help message and exit
