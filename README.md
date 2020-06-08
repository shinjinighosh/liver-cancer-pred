# liver-cancer-pred

Playground for trying to predict liver cancer

### How to run the DNN model

    git clone git@github.com:shinjinighosh/liver-cancer-pred.git
    cd liver-cancer-pred
    python NN_playground.py

If `python NN_playground.py` does not work, it could be because of multiple python versions or virtual environments being around on your system. Try `python3 NN_playground.py`!

You can look at the model loss and accuracies on both training and validation sets in `Loss.png` and `Accuracy.png` respectively. The confusion matrix is plotted prettily at `Confusion_matrix.png`.

The model metrics are printed at the end. Feel free to play around with the model!

### How to run the MLP models

    git clone git@github.com:shinjinighosh/liver-cancer-pred.git
    cd liver-cancer-pred
    python MLP_playground.py

If `python MLP_playground.py` does not work, it could be because of multiple python versions or virtual environments being around on your system. Try `python3 MLP_playground.py`!

Currently, maximum iterations is set to 15,000. Feel free to increase it if some models fail to converge!
The best model out of those compared, along with its parameters and loss and accuracy values are printed at the end.
