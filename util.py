import pandas as pd
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
import pickle



gendered_pronoun_df = pd.read_csv('test_stage_1.tsv', delimiter='\t')

#trasform the sentence in a list of word
gendered_pronoun_df['Text'] = gendered_pronoun_df['Text'].apply(lambda x: x.split())


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


elmo = ElmoEmbedder(options_file, weight_file)
embeddings = elmo.embed_sentence(gendered_pronoun_df.Text[1])
#embeddings = elmo.embed_batch(gendered_pronoun_df.Text)

print (embeddings)
