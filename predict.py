from core import process_text
import pickle
import pandas as pd
import json
import ast
import numpy as np
import sys

# load the model and category names from file
RF_pipeline, categories = pickle.load(open('pipeline.pickle', 'rb'))

# load prediction file
with open(sys.argv[1], 'r') as test_file:
    dict_test = json.load(test_file)

# convert to dataframe
test = pd.io.json.json_normalize(dict_test)

y_test = process_text(test)['content.fullTextHtml']

# predict probabilities using trained model
predictionRF_prob = RF_pipeline.predict_proba(y_test)

# load taxonomy mappings
file = 'taxonomy_mappings.json'
with open(file) as taxonomy_file:
    taxonomy_mappings = ast.literal_eval(taxonomy_file.read())

# find indices (with respect to the full list) of categories used for training
reverse_mappings = {name:num for num,name in taxonomy_mappings.items()}
cat_indices = np.array([int(reverse_mappings[cat]) for cat in categories])

# create list of lists of probabilities for each text
probas = []
for text_num in range(len(test)):
    prob_list = []
    col_index = 0
    for tax_num in range(len(taxonomy_mappings)):
        if tax_num in cat_indices:
            prob_list.append(predictionRF_prob[text_num][col_index])
            col_index += 1
        else:
            prob_list.append(0)
    probas.append(prob_list)

# save probabilities to file
with open('probas.json', 'w') as f:
    json.dump(probas, f)
