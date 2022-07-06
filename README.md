# Saturation Methods
This is code that accompanies a paper titled "Finding democracy in big data: word-embedding-based document retrieval"


## Dependencies
- Python 3.9
- requirements.txt attached to the project

## Dataset

The dataset used for the final validation can be found at https://osf.io/rk6pc/

## Scripts

The raw data used for this project has not been released yet. The dataset mentioned in the last section  
constitutes a subset of the raw data, necessary to replicate the validation analyses, in order to do that  
proceed from the 4th step of the list below. If, on the other hand you plan to replicate the analysis  
using a different dataset, proceed from the 1st step.

To recreate the project's work do the following: 
1. Run the averaging.py script to generate the document embeddings
2. Run the saturation.py script to saturate the corpus using the five different saturation methods outlined in the paper
3. Run the hollowing_out.py script to generate subcorpora for training
4. Run the model.py script to train the models
5. Use the model_results.py script to calculate the model results
6. Run the saturate.py script to generate model topic saturations for each of the subcorpora
7. Run the correlate_subsets.py script to generate the correlation matrix
8. Run the venn_diagram.py script to generate the Venn diagram

## Implementation

The accompanying implementation script is meant to be a simple implementation of the best performing saturation method to be used in research by social scientists from different walks of life.
