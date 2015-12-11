import numpy as np
import matplotlib.pyplot as plt

def analyze(predictions, reality):
	index_to_expression = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	guess_distribution_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}

	for i in range(len(reality)):
		new_list = guess_distribution_dict[reality[i]]
		new_list.append(predictions[i])
		guess_distribution_dict[reality[i]] = new_list

	for i in range(7):
		if len(guess_distribution_dict[i]) > 0:
			plt.subplot(2,4,i+1)
			plt.title('Truth: ' + index_to_expression[i])
			plt.xlabel('Prediction')
			plt.ylabel('# Occurances')
			plt.xlim(1, 7, 1)
			plt.hist(guess_distribution_dict[i], bins = range(8))
			plt.xticks(range(7),index_to_expression,rotation=90, rotation_mode="anchor", ha="right")

			plt.tight_layout()

		percent_correct = 100.0*(float(guess_distribution_dict[i].count(i))/len(guess_distribution_dict[i]))
		print('Percent correct -- ' + index_to_expression[i] + ' ' + str(percent_correct))

	plt.show()



#analyze([1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4],[1,1,1,1,1,2,4,4,2,3,1,1,6,3,0,0,4])