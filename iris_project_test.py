import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TestIrisPipeline(unittest.TestCase):
    def __init__(self):
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names

    def test_data_loading(self):
        self.assertEqual(self.X.shape, (150, 4)) # 150 Samples, 4 Features
        self.assertEqual(self.y.shape, (150,))
        self.assertEqual(len(self.feature_names), 4) # 4 Features sepal/petal width & length
        self.assertEqual(len(self.target_names), 3) # target names are the 3 species

    def test_missing_and_duplicates(self): # Bad practice because unit testing should not include an "and"
        data = pd.DataFrame(self.X, columns= self.feature_names)
        self.assertTrue(data.isnull().sum().sum() == 0) # Sum columns and rows

        duplicate_count = data.duplicated().sum()

        if duplicate_count >= 0:
            print(f"Number of duplicate rows: {duplicate_count}")
            data.drop_duplicates(inplace=True)

        self.assertEqual(data.duplicated().sum(), 0)
    
    def test_data_splitting(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), 120)
        self.assertEqual(len(X_test), 30)

    def test_scaling(self):
        X_train, X_test, _, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)

        self.assertAlmostEqual(X_train_scaled.mean(), 0, delta=0.1) # This data should be approx 0
        self.assertAlmostEqual(X_train_scaled.std(), 1, delta=0.1) # Should be approx. 1
    
    def test_model_training_and_evaluation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        #Test the model's accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.9) # 90% or more

        # Test confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.assertEqual(cm.shape, (3,3))

if __name__ == '__main__':
    unittest.main()

