import unittest 
import joblib
from sklearn.ensemble import RandomForestclassifier

class TestModelTraining(unittest.Testcase):
	def test_model_training(self):
		model= joblib.load('model/iris_model.pkl')
		self.assertIsInstance(model, RandomForestClassifier)
		self.assertGreaterEqual(len(model.feature_importances_),4)
if__name__=='__main__':
	unittest.main()