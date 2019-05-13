# CS466_Final_Project
This contains an implementation of Convolutional Neural Network for modeling Gene Regulation

Expression_Model.py contains the Convolutional Neural Network model of gene expression. This class takes the enhancer sequence, ground truth expressions and transcription factor profiles as a sequence object (Seq.py). Sequence object is constructed using the three files in the input folder that contain the data used in this project. Also, the sequence object has a method that supports stochastic gradient descent optimization.

Seq.py: Contains the code for the sequence class.

main.py: is used to train the model according to the input hyper parameter settings. It takes the following flags:

--lr	Learning rate of gradient descent
--bs	Batch size used for training
--ns	Total number of training steps (total number of batches used for training)
--ds	Dropout rate of the first layer of convolutional filters
--dm	Dropout rate of the second layer of convolutional filters
--rm	L2 regularization scale for the parameters of the first layer of convolutional filters
--rc	L2 regularization scale for the parameters of the second layer of convolutional filters
--rnn	L2 regularization scale for the final fully connected layer
--otf	Output file name (path) for writing tensorflow parameters and checkpoint
--otr	Output file name (path) for writing train errors
--ote 	Output file name (path) for writing test errors after training
--ov 	Output file name (path) for writing validation errors
--op	Output file name (path) for writing trained parameters dictionary
