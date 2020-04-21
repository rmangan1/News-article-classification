This project consists of a pipeline for preprocessing the training data (news articles with taxonomy labels), training a classification model and then making predictions given new articles. DVC is used to create the pipeline (linking preprocess.py -> train.py -> predict.py).

core.py contains the process_text function which is used in multiple files. It processes the news articles (removing HTML, white spaces, etc. and stemming the words).

preprocess.py takes a json file containing news articles along with corresponding section names and taxonomy probabilities (generated using another algorithm). It organizes the data into a pandas dataframe and cleans up the news articles. The section names have been created manually. The section names are used to supplement the taxonomy labels. For example, if the section name is "lifestyle", we assign a probability of 1 to the "lifestyle" taxonomy. Some section names have no obvious taxonomy counterpart. In this case, I found the taxonomies which correlated the most with those section names (e.g. the "bi -> strategy" section is correlated with the "economy, business and finance" taxonomy). If a news article had one of those section names, and its taxonomy probability for a correlated taxonomy was greater than 0.3, then a probability of 1 was assigned for that label. In this way, section names are used to "strengthen" the taxonomy labels. Finally, any probabilities greater than 0.5 are set to 1, while those less than 0.5 are set to 0.

train.py
Here I take the training data from the previous step and train a Random Forest Classifier, first applying tf-idf vectorization.

predict.py
This file loads the model and used it to predict the taxonomies (return a probability for each taxonomy), given one or more news articles as input.

create_env.sh
This contains the commands to create the necessary conda environment.

The pipeline can be run with these commands:
dvc init --no-scm
source create_env.sh
dvc repro predict.dvc

The train data, test data and taxonomy mappings are not included in this folder.
