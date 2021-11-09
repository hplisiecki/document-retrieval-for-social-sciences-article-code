# Saturation Methods
This is code that accompanies a paper titled "Extracting general topics from large text corpora"


## Dependencies
- Python 3.9
- requirements.txt attached to the project

## Dataset

## Scripts

To recreate the project's work do the following:
1. Run the averaging.py script to generate the document embeddings
2. Run the saturation.py script to saturate the corpus using the five different saturation methods outlined in the paper
3. Run the hollowing_out.py script to generate subcorpora for training
4. Run the model.py script to train the models
5. Use the model_results.py script to calculate the model results
6. Run the saturate.py script to generate model topic saturations for each of the subcorpora
7. Run the correlate_subsets.py script to generate the correlation matrix
