## Binary classification Sclerotic Vs. Non sclerotic glomeruli
### Data Format 
Create folder named data and create subdirectories for each class separately
### Training and Evaluation
Run the train_eval.ipynb file sequentially to train and test the model.
If you want to only test the model load the data and run the last cell of the file.
Model is trained for 20 epochs and the model is stored in .pth format.
Model is run on avaialble PyTorch2.2.0 environment on HPG.
ResNet18 mode with pretrained imagenet weights are used to train the model.
There is a train_eval.py file as well feel free to run it as well.