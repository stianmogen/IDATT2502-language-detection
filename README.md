# IDATT2502-language-detection
 
This repository is for the final project in the course IDATT2502-Machine-Learning.

The project contains the following directories:

- mnb
- rnn
- test-client
- utils
- input

### MNB (Multinomial Naive Bayes) 

The MNB model is trained in the file named mnb-test-py. By running this you can create the models with your preffered n-grams. (Note, the models become several gigabytes large for n >= 3. If you want to test these yourself, they have to be created locally). 

The models are stored in the "out" directory. When running the model, each new model and belonging vectorized, will get its own directory corresponding to the n-values and analyzer.

### RNN (Reccurent Neural Network) 

This directory stored the RNN models, LSTM and GRU. 
- The RNN-model is trained in rnn-train, which uses the rnn-model class for creating the GRU and LSTM models. 
- The dictionary class is used for RNN models in training and prediciton
- The results from these models are stored in the "out" directory, and with their corresponding model-type, direction and hidden-size value.
Additionally a plot with the accuracy and loss for the given model will also be stored here.  

### Test-Client

A test-client lets the user load their preffered model in the command line. When it is loaded, the user may write a sentence for prediciton.
The currently loaded model will then predict the language of the sentence. 

### Utils 

The utils directory contains necessary classes for the models in the project.
- Batch Generator: Loads data in batches to ensure limitation of memory usage when training.
- Split Dataset: Splits the training, validation and test-data into by users preffered ration.
- Dataloader: Loads the dataset for usage in models, here you can choose specific languages for testing purposes.
- Model Validator: Validates the accuracy of the model.
- Confusion: Provides the bigges amount of mistakes when predicting, for use in cunfusion matrices. 

### Input

The input directory contains the original dataset witht the corresponding urls and labels. The dataset is the "WiLI-2018, and was provided [here](https://zenodo.org/record/841984#.YZ-Hvr3MIq0).

The dataset used by the models are saved in the "dataset" directory. The dataset split into the preffered training, validation and test ratio is saved in files ending with "split.txt".
