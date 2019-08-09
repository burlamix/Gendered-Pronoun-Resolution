import numpy as np 
import pandas as pd 

import os,sys

import zipfile
import sys
import time
import wget


#bad trick for import
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from extract_features import *
import modeling 
import tensorflow as tf


def compute_offset_no_spaces(text, offset):
	count = 0
	for pos in range(offset):
		if text[pos] != " ": count +=1
	return count

def count_chars_no_special(text):
	count = 0
	special_char_list = ["#"]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count

def count_length_no_special(text):
	count = 0
	special_char_list = ["#", " "]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count

def run_bert(data):
	'''
	Runs a forward propagation of BERT on input text, extracting contextual word embeddings
	Input: data, a pandas DataFrame containing the information in one of the GAP files

	Output: emb, a pandas DataFrame containing contextual embeddings for the words A, B and Pronoun. Each embedding is a numpy array of shape (768)
	columns: "emb_A": the embedding for word A
	         "emb_B": the embedding for word B
	         "emb_P": the embedding for the pronoun
	         "label": the answer to the coreference problem: "A", "B" or "NEITHER"
	'''
    # From the current file, take the text only, and write it in a file which will be passed to BERT
	text = data["Text"]
	text.to_csv("input.txt", index = False, header = False)

    # The script extract_features.py runs forward propagation through BERT, and writes the output in the file output.jsonl
    # I'm lazy, so I'm only saving the output of the last layer. Feel free to change --layers = -1 to save the output of other layers.

	os.system("python3 hltproject/dataset_utils/extract_features.py \
	  --input_file=input.txt \
	  --output_file=output.jsonl \
	  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
	  --bert_config_file=hltproject/utils/uncased_L-12_H-768_A-12/bert_config.json \
	  --init_checkpoint=hltproject/utils/uncased_L-12_H-768_A-12/bert_model.ckpt \
	  --layers=-1 \
	  --max_seq_length=256 \
	  --batch_size=8")


	bert_output = pd.read_json("output.jsonl", lines = True)


	os.system("rm output.jsonl")
	os.system("rm input.txt")

	index = data.index
	columns = ["emb_A", "emb_B", "emb_P", "label"]
	emb = pd.DataFrame(index = index, columns = columns)
	emb.index.name = "ID"

	for i in range(len(data)): # For each line in the data file
		# get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
		P = data.loc[i,"Pronoun"].lower()
		A = data.loc[i,"A"].lower()
		B = data.loc[i,"B"].lower()

		# For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
		P_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"Pronoun-offset"])
		A_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"A-offset"])
		B_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"B-offset"])
		# Figure out the length of A, B, not counting spaces or special characters
		A_length = count_length_no_special(A)
		B_length = count_length_no_special(B)

		# Initialize embeddings with zeros
		emb_A = np.zeros(768)
		emb_B = np.zeros(768)
		emb_P = np.zeros(768)

		# Initialize counts
		count_chars = 0
		cnt_A, cnt_B, cnt_P = 0, 0, 0

		features = pd.DataFrame(bert_output.loc[i,"features"]) # Get the BERT embeddings for the current line in the data file
		for j in range(2,len(features)):  # Iterate over the BERT tokens for the current line; we skip over the first 2 tokens, which don't correspond to words
			token = features.loc[j,"token"]

			# See if the character count until the current token matches the offset of any of the 3 target words
			if count_chars  == P_offset: 
				# print(token)
				emb_P += np.array(features.loc[j,"layers"][0]['values'])
				cnt_P += 1
			if count_chars in range(A_offset, A_offset + A_length): 
				# print(token)
				emb_A += np.array(features.loc[j,"layers"][0]['values'])
				cnt_A +=1
			if count_chars in range(B_offset, B_offset + B_length): 
				# print(token)
				emb_B += np.array(features.loc[j,"layers"][0]['values'])
				cnt_B +=1								
			# Update the character count
			count_chars += count_length_no_special(token)
		# Taking the average between tokens in the span of A or B, so divide the current value by the count	
		emb_A /= cnt_A
		emb_B /= cnt_B

		# Work out the label of the current piece of text
		label = "Neither"
		if (data.loc[i,"A-coref"] == True):
			label = "A"
		if (data.loc[i,"B-coref"] == True):
			label = "B"

		# Put everything together in emb
		emb.iloc[i] = [emb_A, emb_B, emb_P, label]

	return emb

def compute_bert_embeddings_brutal ( input_fname, output_fname ):

	print("------------------\n\n\n\n\n\n")
	print(os.getcwd())
	print(input_fname)
	
	test_data = pd.read_csv(input_fname, sep = '\t')
	test_emb = run_bert(test_data)
	test_emb.to_json(output_fname, orient = 'columns')

	'''
		0 ID                                                           test-9
		1 Text              On June 4, 1973 at the Felt Forum, Madison Squ...
		2 Pronoun                                                          he
		3 Pronoun-offset                                                  227
		4 A                                                            Malave
		5 A-offset                                                        124
		6 A-coref                                                        True
		7 B                                                       Greg Joiner
		8 B-offset                                                        169
		9 B-coref                                                       False
		10 URL                       http://en.wikipedia.org/wiki/Edwin_Malave
	'''
def compute_bert_embeddings (input_fname, output_fname):

	test_data = pd.read_csv(input_fname, sep = '\t')
	for index, row in test_data.iterrows():
		size_p = len(row[2])
		size_a = len(row[4])
		size_b = len(row[7])

		a_list= [(row[3],"pp119",size_p),(row[5],"a19",size_a),(row[8],"b19",size_b)]

		#sorted(a_list, key=lambda x: x[0])
		a_list.sort(key=lambda x:x[0],reverse=True)

		#print(a_list)

		#print(row)
		#print(row[1])
		#print(row[1])

		mod= row[1][:a_list[0][0]] + " "+ a_list[0][1] +" "+  row[1][a_list[0][0] +a_list[0][2]:] 
		mod= mod[:a_list[1][0]]    + " "+ a_list[1][1] +" "+ mod[a_list[1][0]    +a_list[1][2]:] 
		mod= mod[:a_list[2][0]]    + " "+ a_list[2][1] +" "+ mod[a_list[2][0]    +a_list[2][2]:] 
		
		mod = " ".join(mod.split())
		mod = ": " + mod
		
		mod = mod.replace(" "+row[4]+" "," a1 ")
		mod = mod.replace(" "+row[7]+" "," b1 ")
		mod = mod.replace(" "+row[2]+" "," pp11 ")

		#print(mod)
		#print("\n")

		test_data.at[index,"Text"]=mod
		
		new_a_off = mod.find("a19")
		new_b_off = mod.find("b19")
		new_p_off = mod.find("pp119")

		test_data.at[index,"A-offset"]=new_a_off
		test_data.at[index,"B-offset"]=new_b_off
		test_data.at[index,"Pronoun-offset"]=new_p_off


	text = test_data["Text"]


	text.to_csv("input.txt", index = False, header = False)

	print(os.getcwd())
	print("\n\n")
	extract_bert_feature(input_file="input.txt",vocab_file="hltproject/utils/uncased_L-12_H-768_A-12/vocab.txt",
		bert_config_file="hltproject/utils/uncased_L-12_H-768_A-12/bert_config.json",
			init_checkpoint="hltproject/utils/uncased_L-12_H-768_A-12/bert_model.ckpt",output_file=output_fname,
            	layers="-1",do_lower_case=True,master=None,num_tpu_cores=True,max_seq_length=500,
            		use_tpu=False,use_one_hot_embeddings=False,batch_size=8,info_data=test_data)


if __name__ == "__main__":
    output_fname = sys.argv[2] + '/' + os.path.basename (sys.argv[1]) if len(sys.argv)>=3 else sys.argv[1]
    compute_embeddings ( sys.argv[1], output_fname )

#compute_b("../datasets/gap-light.tsv","output_x.jsonl")

#print("\n\n\n\n\n\n---------END---------\n\n\n\n\n\n")
