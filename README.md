# GoodReadBooksReviews
Project Members: Kaydn Brady and Michael Miner

In this repository we have main.py where we perform the bulk of our neural network building and preprocessing. In the file FilterReview.py we provide the support functions needed to perform filtering on the input dataset such as removeing unwanted character, words, and symbols.

The final Sequential model structure is as follows:

Embedding()

Conv1D(filters=300, kernel_size=1)

Conv1D(filters=30, kernel_size=4)

MaxPooling1D(pool_size=2))

Bidirectional(LSTM(32))

Dropout(0.4))

Dense(16)

Dense(6)

The final accuracy of this model on the GoodReads Kaggle Competition dataset was at 56%.
