# Importing libraries 
import sys
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import time

# Download resource 'punkt'
nltk.download('punkt')

# Loading the Spacy English model
nlp = spacy.load('en_core_web_md')

st_time = time.time()


# Measuring similarity index
def cal_sim(sent_list):
    sim_array = []
    for i in range(len(sent_list) - 1):
        query_doc = nlp(sent_list[i])
        search_doc = nlp(sent_list[i + 1])
        q_doc_stop = nlp(' '.join([str(t) for t in query_doc if not t.is_stop]))
        s_doc_stop = nlp(' '.join([str(t) for t in search_doc if not t.is_stop]))
        sim = q_doc_stop.similarity(s_doc_stop)
        sim_array.append(sim)
    return sim_array


# processing input arguements
input_file = sys.argv[1]
output_file = sys.argv[2]

try:
    with open(input_file, 'r', encoding='utf-8') as file:
        p = file.read()
except IOError:
    print("Error in reading the file:", input_file)

sent_list = []
if len(p.split('\n')) >= 2:
    paras = p.split('\n')
    for each in paras:
        sent_list.extend(sent_tokenize(each))
else:
    sent_list = sent_tokenize(p)

num_sents = len(sent_list)
new_sent_list = []
interval = 30
for i in range(0, num_sents, interval):
    new_sent_list.append(' '.join(sent_list[i: i + interval]))

sim_array = cal_sim(new_sent_list)

# Calculating threshold for similarity measure                             
thresh = np.mean(sim_array)

# Calculating segments
segments = new_sent_list[0]

for i in range(len(sim_array)):
    if sim_array[i] > thresh:
        segments = segments + '' + new_sent_list[i + 1]
    else:
        segments = segments + '\n \n -------------------segment-------------- \n \n' + new_sent_list[i + 1]

f = open(output_file, "w")
f.write(segments)
f.close()

print("Ended.....{}".format(time.time() - st_time))
