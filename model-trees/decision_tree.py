import pandas as pd
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import warnings
import math


class Node:
	"""
	A Node class for the ID3 decision tree. This is not to be used directly,
	please use the DecisionTree class instead.
	"""
	def __init__(self, data: pd.DataFrame, Y_attr: str, split_con='None'):
		"""
		Creates a new Node instance.
		
		Args:
		-----
		data: The data to be contained in this node
		Y_attr: The string representing the Y attribute
		split_con: The attribute that this data was split on
		"""
		self.data = data
		self.y_attr = Y_attr
		self.split_condition = split_con
		self.children = []
		self.is_learner = False
		self.learner = None

	def get_entropy(self, data: pd.DataFrame) -> float:
		"""
		Gets the entropy of the data.
		
		Args:
		-----
		data: The data to find the entropy of
		"""
		cnt = Counter(data[self.y_attr])
		probs = [x / len(data.index) for x in cnt.values()]
		return sum([-p * math.log(p, 2) for p in probs])

	def get_splitter(self):
		"""
		Checks if the majority vote beats learners on data.

		:return: None if majority vote wins; else returns a learner.
		"""
		learners = [KNeighborsClassifier(), SVC(), GaussianNB(), LogisticRegression()]
		best_learner = DummyClassifier(strategy='most_frequent')

		data_x = self.data[[col for col in self.data.columns if col != self.y_attr]]
		data_y = self.data[self.y_attr]

		X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
		best_learner.fit(X_train, y_train)
		preds = best_learner.predict(X_test)
		baseline_score = recall_score(y_test, preds)
		print('Baseline score is', baseline_score)
		best_score = baseline_score

		for learner in learners:
			learner.fit(X_train, y_train)
			preds = learner.predict(X_test)
			score = recall_score(y_test, preds)

			if score > best_score:
				best_learner = learner
				best_score = score

		if best_score > baseline_score:
			print('Splitter is a learner:', best_learner.__class__.__name__, 'with score', best_score)
			return best_learner
		else:
			return None
	
	def get_split_condition(self) -> tuple:
		"""
		Returns the attribute, which on splitting by yields the highest
		information gain.
		"""
		best_split = None
		best_midpoint = 0.
		info_gain = 0
		par_entropy = self.get_entropy(self.data)

		for col in self.data.columns:
			if col != self.y_attr:
				vals = sorted(self.data[col])
				for pair in zip(vals, vals[1:]):
					midpoint = np.mean(pair)

					groups = [df[df[col] <= midpoint], df[df[col] > midpoint]]

					for group in groups:
						entropy = self.get_entropy(group)
						cur_info_gain = par_entropy - len(group.index) / len(self.data.index) * entropy
						if cur_info_gain > info_gain:
							best_split = col
							info_gain = cur_info_gain
							best_midpoint = midpoint

		return best_split, best_midpoint, info_gain
	
	def split(self, verbose=True) -> None:
		"""
		Splits the data in the current node by the best split condition.
		
		Args:
		-----
		verbose: Prints debug information
		"""
		split, midpoint, info_gain = self.get_split_condition()

		if verbose:
			print('Splitting on', split, 'at', midpoint, 'with information gain', info_gain)
		
		# Split the data by the condition
		groups = [df[df[split] <= midpoint], df[df[split] > midpoint]]

		if verbose:
			print('Children:')

		for group in groups:
			# Remove the split condition column and create a node from the
			# resulting dataset
			group.drop(split, axis=1, inplace=True)

			if verbose:
				print('---------\n', group)

			n = Node(group, self.y_attr, split)
			self.children.append(n)


class DecisionTree(Node):
	"""
	A DecisionTree class that implements the ID3 algorithm using the Node
	class.	
	"""
	def __init__(self, data: pd.DataFrame, y: str):
		"""
		Creates a DecisionTree object.
		
		Args:
		-----
		data: The data for the current node
		y: The output attribute
		"""
		super().__init__(data, y)

	def fit(self) -> None:
		"""
		Creates the full decision tree from the current data.
		"""
		stack = [self]

		while len(stack) > 0:
			node = stack.pop()
			
			# If entropy is 0, then stop splitting.
			if node.get_entropy(node.data) > 0:
				splitter = node.get_splitter()
				if splitter is None:
					node.split(verbose=False)
					for child in node.children:
						stack.append(child)
				else:
					node.is_learner = True
					node.learner = splitter

	def print(self) -> None:
		"""
		Prints the decision tree nodes' data.
		"""
		level = 0
		stack = [(level, self)]

		while len(stack) > 0:
			level, node = stack.pop()
			print('\nLevel', level, 'Split condition:', node.split_condition, '\n-----------')

			if node.is_learner:
				print(node.get_splitter())
				continue
			for child in node.children:
				stack.append((level + 1, child))

	def predict(self, samples: pd.DataFrame) -> str:
		"""
		Returns the class label for the given sample.
		
		Args:
		-----
		sample: A DataFrame containing a single sample to predict on
		"""
		preds = []
		for _, sample in samples.iterrows():
			print('Processing sample', _)
			node = self

			while len(node.children) > 0:
				print(node)
				split, midpoint, info_gain = node.get_split_condition()

				# Check if this child has the right value of the splitting
				# condition. If not, try another child.
				if sample[split] <= midpoint:
					node = node.children[0]
				else:
					node = node.children[1]

			if node.children is None:
				preds.append(node.get_splitter().predict(self.data[[col for col in self.data.columns if col != self.y_attr]]))
			else:
				preds.append(list(node.data[self.y_attr])[0])

		return preds


if __name__ == '__main__':
	warnings.filterwarnings('ignore')
	df = pd.read_csv('ivy-1.4.csv')
	df['bug'] = df['bug'].apply(lambda x: 0 if x == 0 else 1)
	root = DecisionTree(df, 'bug')
	root.fit()
	root.print()
	print('-------------------------\nPredicting for data:')
	test = pd.read_csv('ivy-2.0.csv')
	test['bug'] = test['bug'].apply(lambda x: 0 if x == 0 else 1)
	y_test = np.copy(test['bug'])
	print(y_test)
	test.drop('bug', inplace=True, axis=1)
	preds = root.predict(test)
	print(np.array(preds).squeeze())
	print('Score:', recall_score(y_test, preds))