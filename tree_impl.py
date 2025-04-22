import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.model_selection import train_test_split

dataset_path = "mushroom_dataset.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError("Dataset not found.")

df = pd.read_csv(dataset_path)

# Create a dictionary to store encoders (if you need to reverse the encoding later)
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# print("Original Label Encoded Data Sample:")
# print(df.head())

# Prepare Data for Tree Building
target_col = 'class'
feature_names = [col for col in df.columns if col != target_col]
X = df[feature_names].values  # Features as numpy array
y = df[target_col].values     # Target labels as numpy array

# Helper Functions: Entropy, Information Gain & Majority Class


def entropy(y):
    """Compute the entropy of a label distribution."""
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    # Adding a small epsilon to avoid log2(0) issues.
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def information_gain(X, y, feature_index):
    """
    Compute the information gain obtained by splitting on the column at feature_index.
    X : 2D numpy array with shape (n_samples, n_features) from the current sub-problem.
    y : 1D numpy array of labels corresponding to X.
    feature_index: integer index in the current feature set.
    """
    total_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    weighted_entropy = 0
    for i, val in enumerate(values):
        subset_y = y[X[:, feature_index] == val]
        weighted_entropy += counts[i] / len(y) * entropy(subset_y)
    return total_entropy - weighted_entropy

def most_common(y):
    """Return the most common label in y."""
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]
# Decision Tree Implementation (ID3)
class Node:
    def __init__(self, feature_index=None, value=None, branches=None, is_leaf=False):
        """
        A Node in the decision tree.
          - feature_index: index (in the current feature array) of the feature used for splitting.
          - value: if internal, this is the feature (name) used to split; if leaf, the predicted label.
          - branches: dictionary mapping each feature value to a child Node.
          - is_leaf: flag indicating whether this node is a leaf.
        """
        self.feature_index = feature_index
        self.value = value
        self.branches = branches if branches is not None else {}
        self.is_leaf = is_leaf

class DecisionTreeClassifierCustom:
    def __init__(self, max_depth=None):
        self.tree = None
        self.original_feature_names = None
        self.max_depth=max_depth
    def fit(self, X, y, feature_names):
        """
        Build the decision tree using the training data.
          X: 2D array of features.
          y: 1D array of labels.
          feature_names: list of feature names corresponding to the columns of X.
        """
        self.original_feature_names = feature_names.copy()
        self.tree = self._build_tree(X, y, feature_names, depth=0)

    def _build_tree(self, X, y, feature_names, depth):
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(is_leaf=True, value=most_common(y))
        # If only one class remains, return a leaf node with that class.
        if len(np.unique(y)) == 1:
            return Node(is_leaf=True, value=y[0])
        # If no more features to split on, return a leaf node with the majority class.
        if len(feature_names) == 0:
            return Node(is_leaf=True, value=most_common(y))

        # Compute information gain for each feature in the current feature set.
        gains = [information_gain(X, y, i) for i in range(len(feature_names))]
        best_feature_index = np.argmax(gains)
        best_feature_gain = gains[best_feature_index]
        best_feature_name = feature_names[best_feature_index]

        # If the best gain is zero, no further splitting is useful.
        if best_feature_gain == 0:
            return Node(is_leaf=True, value=most_common(y))

        # Create an internal node that splits on the best feature.
        node = Node(feature_index=best_feature_index, value=best_feature_name, is_leaf=False)
        feature_values = np.unique(X[:, best_feature_index])

        # For each value of the best feature, build a subtree.
        for value in feature_values:
            indices = np.where(X[:, best_feature_index] == value)[0]
            X_subset = X[indices, :]
            y_subset = y[indices]

            # Remove the best feature column from X_subset.
            X_subset = np.delete(X_subset, best_feature_index, axis=1)
            # Remove the corresponding feature name.
            new_feature_names = feature_names.copy()
            new_feature_names.pop(best_feature_index)

            # Recursively build the subtree and assign it to the branch for this feature value.
            node.branches[value] = self._build_tree(X_subset, y_subset, new_feature_names, depth + 1)
        return node

    def predict_sample(self, x, tree, feature_names):
        """
        Recursively traverse the tree to predict the label for a single sample.
          x: 1D numpy array representing one sample's feature values.
          tree: current node (starting with the root).
          feature_names: list of feature names corresponding to x.
        """
        if tree.is_leaf:
            return tree.value
        index = tree.feature_index
        feature_val = x[index]
        if feature_val in tree.branches:
            subtree = tree.branches[feature_val]
        else:
            # If a feature value is not present in the branches, fall back to majority class among branches.
            predictions = [self._get_majority_class(sub) for sub in tree.branches.values()]
            return max(set(predictions), key=predictions.count)
        # Remove the used feature from x and feature_names for the recursive call.
        new_x = np.delete(x, index)
        new_feature_names = feature_names.copy()
        new_feature_names.pop(index)
        return self.predict_sample(new_x, subtree, new_feature_names)

    def predict(self, X):
        """
        Predict labels for a set of instances.
          X: 2D numpy array of instances.
        """
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x, self.tree, self.original_feature_names.copy()))
        return np.array(predictions)

    def _get_majority_class(self, node):
        """Recursively obtain the majority class from the subtree rooted at node."""
        if node.is_leaf:
            return node.value
        values = [self._get_majority_class(branch) for branch in node.branches.values()]
        return max(set(values), key=values.count)
def print_tree(node, feature_names, indent=""):
    """
    Recursively prints the structure of the decision tree.
      node: current Node.
      feature_names: current list of feature names.
      indent: string used for indentation (for readability).
    """
    if node.is_leaf:
        # For leaves, print the predicted class.
        print(indent + "Leaf:", node.value)
    else:
        # For internal nodes, print the feature used.
        print(indent + "Feature:", node.value)
        for branch_val, subtree in node.branches.items():
            print(indent + f" -> If {node.value} == {branch_val}:")
            # Copy feature_names and remove the used feature.
            new_feature_names = feature_names.copy()
            if node.value in new_feature_names:
                new_feature_names.remove(node.value)
            print_tree(subtree, new_feature_names, indent + "    ")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree_classifier = DecisionTreeClassifierCustom()
tree_classifier.fit(X_train, y_train, feature_names.copy())


#print("\nDecision Tree Structure:")
#print_tree(tree_classifier.tree, feature_names.copy())
train_preds = tree_classifier.predict(X_train)
test_preds = tree_classifier.predict(X_test)

train_accuracy = np.mean(train_preds == y_train)
test_accuracy = np.mean(test_preds == y_test)



print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))





# after you build the label_encoders dict:
#    label_encoders: Dict[str, LabelEncoder]
# make it importable at top‐level
__all__ = ["DecisionTreeClassifierCustom", "print_tree", "label_encoders", "feature_names"]

