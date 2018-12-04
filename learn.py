import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle

class LearnModel:
	def __init__(self, csv_file, model_file):
		print("##### Learning Model #####")
		self.csv_file = csv_file
		self.model_file = model_file

	def train(self):
		data = pd.read_csv(self.csv_file)
		x = data[['ACYC','AFREQ','C0res%','C10res%','C1res%','C2res%','C3res%','C4res%','C5res%','C6res%','C7res%','C8res%','C9res%','EXEC','FREQ','INST','INSTnom','INSTnom%','IPC','L2HIT','L2MISS','L2MPI','L3HIT','L3MISS','L3MPI','PhysIPC','PhysIPC%','Proc Energy (Joules)','READ','TIME(ticks)','WRITE']]
		y = data['Slowdown']
	
		x_train, self.x_test, y_train, self.y_test = train_test_split(x, y, test_size=0.4, random_state=101)
		model = LinearRegression()
		model.fit(x_train,y_train)

		# save the model to disk
		pickle.dump(model, open(self.model_file, 'wb'))
 
	def measure(self): 
		# load the model from disk
		loaded_model = pickle.load(open(self.model_file, 'rb'))
		y_pred = loaded_model.predict(self.x_test)
		print(np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)))
		print(r2_score(self.y_test, y_pred)) 
	
if __name__ == "__main__":
	csv_file = sys.argv[1]
	model_file = 'model.pkl'
	lm = LearnModel(csv_file, model_file)
	lm.train()
	# Comment the following line in the actual code
	lm.measure()
