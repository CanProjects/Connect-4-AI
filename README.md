# Connect4

A very strong machine learning based approach to Connect4, incorporating no prior game knowledge. Inspired by deepmind's alphazero. The implementation combines multiple different approaches to game solving, including min-maxing, monte carlo tree search, and machine learning (using a convolutional neural network) to classify/evaluate the strength of positions.

# How to play against it yourself

Open CNNPlayWithSearch.py, run the program. The AI as it is currently built can only play first, but I believe also making it play second could actually be a very easy change. 

To make the AI think shorter/longer you can go into heuristics > treeGen and then change counterL < whatever amount of positions you want the AI to see. The AI will increase amount of positions seen in powers of 7, so be careful with selection of large numbers.



