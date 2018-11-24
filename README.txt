Raunaque Patra	(150050063)
Srijit Dutta	(150070046)
Ankan Sardar	(150050064)
Ayush Raj		(150050042)

##################################################################################################################################


rnalib.py	: contains the network architectures, Environment definations and policy update procedure for DQN.

train.py 	: contains the main algorithm to train & update the policies(RNA_BiLSTM_Policy/RNA_CNN_Policy/tabularpolicy). After conpleting training it save the pytorch model as 'DQN_policy' in the parent directory.(Note existing 'DQN_policy' will be overwitten.). It takes a string argument 'cnn','lstm' or 'dyna'
eg $python3 train.py lstm

solve_one_puzzle.py : used to find the solution for 1 problem, takes as argument either RNA sequence or a target dot-bracket structure. It uses the saved model from train.py

eg. $python3 solve_one_puzzle.py '(((...............)))'
	$python3 solve_one_puzzle.py 'GGGGGAAAACCCCC'


masterTest.py 	: used to run test on the eterna-100 dataset. It uses the saved model from train.py (takes many minutes).

testOnGenerated.py 	: used to run test on the generated datasets of rna length 10,12,15,20,30. It then plots the results for random & the trained model. It uses the saved model from train.py(takes many minutes).

journal.pcbi.1006176.s001.csv	: Eternal 100 dataset

##################################################################################################################################

Two trained models are saved in the folder "savedModels". To use these models in solve_one_puzzle.py, masterTest.py  or testOnGenerated.py copy the required file and paste it in the parent directory and rename it to 'DQN_policy'. (this name is used in all other files)

detailed explanations are in report.pdf