# Connect4

A very strong machine learning based approach to Connect4, incorporating no prior game knowledge. Inspired by deepmind's alphazero. The implementation combines multiple different approaches to game solving, including min-maxing, monte carlo tree search, and machine learning (using a convolutional neural network) to classify/evaluate the strength of positions.

# How to play against it yourself

Open CNNPlayWithSearch.py, run the program. The AI as it is currently built can only play first, but I believe also making it play second could actually be a very easy change. 

To make the AI think shorter/longer you can go into CNNPlayWithSearch.py > heuristics > treeGen and then change counterL < whatever amount of positions you want the AI to see. The AI will increase amount of positions seen in powers of 7, so be careful with selection of large numbers.

# How to have it play against pure monte carlo tree search with no AI.

Open connect4.py, in connect4 > Step > find actionR variable and change it to actionR = monte(self.getState(),4000). You can change 4000 to any number to increase or decrease the strength of the monte carlo tree search player.

#How to train your own network

You need to first generate games. MonteGameGen.py creates games in a multiprocessing way. Currently the game generation is commented out. Uncomment the game generation part, and comment everything after. Generating games should take a long time and should make many files. You can stop the program at any time when you think theres enough games. These games are generated using just monteSearch of depth 1000, which you can change by scrolling up in MonteGameGen.py and changing the number from 1000 to something smaller if you want it to run faster. As a rule of thumb, you want about 200mb of data generated total. Sort these files into draws, wins and losses, then merge the CSV files into files called 0.csv , 1.csv and neg1.csv. Go to the end of 0.csv, 1.csv and neg1.csv and delete the last row which should contain an empty array (I think this is a bug as a result of multiprocessing). Uncomment the second half of MonteGameGen.py and comment the first part that you just ran. This should create a file called final.csv which contains a large dataset of positions and their results. Now open the file Connect4CnnModel.py. This file should take a while, so be patient.

