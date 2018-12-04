import sys
from sklearn.linear_model import LinearRegression
import pickle
import ast

class PredictModel:
	def __init__(self, model_file):
		self.model = pickle.load(open(model_file, 'rb'))

	def predict(self, vec):
		pred_slowdown = self.model.predict(vec)[0]
		return pred_slowdown

def convert(vec_str):
	vec_list = ast.literal_eval(vec_str)
	return vec_list

if __name__ == "__main__":
	model_file = sys.argv[1]
	#inp_vec = [[1.43E+05,1.01,99.9,0,0.121,0,0,0,0,0,0,0,0,1.1,1.01,1.56E+05,2.2,55.1,1.09,0.562,67.5,0.000432,0.223,48.4,0.000309,2.19,54.7,274,20.6,1.77E+04,38.8]]
	inp_vec = convert(sys.argv[2])
	lm = PredictModel(model_file)
	print(lm.predict([inp_vec]))
