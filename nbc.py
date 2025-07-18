import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter, defaultdict
from code import interact

'''
Implementation of the categorical NBC classifier
We assume all input features are categorical
This implementation includes pseudocount logic

Usage:
    nbc_classifier = NBC()

'''
class NBC:
    def __init__(self):
        pass
        # this is a flag to ensure that we don't call predict before train
        # it is set to true iff train ran successfully (control flow reaches end of train method)
        self.train_was_called = False

    '''
    only pass in the training data to this method
    X: pandas.core.frame.DataFrame
        We assume each row of X is entry to test, columns are features
    y: numpy.ndarray
        We assume y is a 1D ndarray whose length is the number of rows of X
    re_train: flag to ensure we don't accidentally call train multiple times
    returns: nothing
    '''
    def train(self, X, y, re_train = False):
        if self.train_was_called:
            if not self.re_train:
                print(
                    "ignoring call to train because train has been called previously on 'this'." + \
                    " To re-train, pass in re_train=True as an argument to train."
                )
                return
        assert len(X.shape) == 2
        self.num_entries, self.num_features = X.shape
        assert y.shape == (self.num_entries, )

        # P(class|features) = P(features|class) × P(class) / P(features)

        self.output_classes = list(set(y))

        self.output_class_counts = Counter(y)

        self.features = list(X.columns)

        self.feature_distinct_values = {ft : set(X[ft]) for ft in self.features}

        # 'e': {'cap-shape': 2, 'odor': 3, ..}
        self.per_class_feature_counts = {}

        # initialize per_class_feature_counts
        # {output_class: {feature: {feature_value_1: 0, feature_value_2: 0, ..}}}
        for o_class in self.output_classes:
            self.per_class_feature_counts[o_class] = \
                {ft : defaultdict(int) for ft in self.features}
            
        # populate per_class_feature_counts
        for i in range(self.num_entries):
            entry = X.iloc[i]
            output_class = y[i]
            for ft, ft_value in entry.items():
                self.per_class_feature_counts[output_class][ft][ft_value] += 1

        self.output_class_probs = {
            o_class: self.output_class_counts[o_class]/self.num_entries \
            for o_class in self.output_classes
        }

        # in per_class_feature_probs, we use pseudocounts to avoid prediction probs collapsing to 0 
        # access is: [output_class][feature][feature_value]
        self.per_class_feature_probs = {}
        # feature_counts is {feature: {feature_value_1: 0, feature_value_2: 0, ..}}
        for o_class, feature_counts in self.per_class_feature_counts.items():
            feature_probs = {}
            # ft_counts is {feature_value_1: 0, feature_value_2: 0, ..}
            for ft, ft_counts in feature_counts.items():
                # assert set(ft_counts.keys()) == set(X[ft]), \
                #     f"feature values {set(X[ft]) - set(ft_counts.keys())} " + \
                #     f"have 0 count for feature {ft} in output class {o_class}"
                num_ft_values = sum(ft_counts.values())
                pseudocount = len(self.feature_distinct_values[ft])
                # interact(local=dict(globals(), **locals()))
                ft_probs = {}
                all_ft_values = self.feature_distinct_values[ft]
                for ft_value in all_ft_values:
                    ft_probs[ft_value] = \
                        (ft_counts[ft_value] + 1)/(num_ft_values + pseudocount)
                    # funnily enough, if we avoid the pseudocount logic, the test accuracy is better?
                    # ft_probs[ft_value] = \
                    #     (ft_counts[ft_value])/(num_ft_values)
                feature_probs[ft] = ft_probs
            self.per_class_feature_probs[o_class] = feature_probs

        # TODO: sanity checks: print warining if any probabilities are zero/close to 0

        self.train_was_called = True

    '''
    helper for predict, predicts output class given one entry of feature values
    returns: a tuple of (prediction class, prediction probability)
    '''
    def predict_one(self, entry):

        '''
        Bayes theorem for us: 
        P(C|features) = P(features|C) x P(C) / P(features)
        where:
            - C represents one particular output class
            - features represents a set of attribute values: think one row of the test dataset
            - P(C|features) is the probability that the set of attribute values belongs to class C
            - P(features|C) is the probability that each attribute takes on the particular 
                value in features, the joint probability of that for all attributes. Because NBC assumes that
                the attributes are independent conditioned on a particular output class, we can just
                multiply P(attribute_i = feature_i|C) for all i attributes

        We won't compute P(features) because it doesn't depend on C, finding the 
        maximum P(features|C) x P(C) across all output classes C is sufficient
        '''

        probs = {}
        for o_class in self.output_classes:
            p = self.output_class_probs[o_class] # P(C)
            for ft, ft_value in entry.items():
                # P(ft_i takes value ft_i_value |C)
                p *= self.per_class_feature_probs[o_class][ft][ft_value]
            probs[o_class] = p
        
        prediction_prob = 0
        prediction_class = ''
        for o_class, p in probs.items():
            if p > prediction_prob:
                prediction_prob = p
                prediction_class = o_class
        # sanity check
        assert prediction_class != ''
        return (prediction_class, prediction_prob)

    '''
    call this after calling train
    X: pandas.core.frame.DataFrame
    We assume each row of X is entry to test, columns are features
    '''
    def predict(self, X):
        if not self.train_was_called:
            print("please call the train method before calling the predict method")
            return

        assert len(X.shape) == 2
        num_test_entries, num_test_features = X.shape
        assert num_test_entries > 0
        # each row is entry to test, columns are features
        assert num_test_features == self.num_features

        predictions = []

        for i in range(num_test_entries):
            entry = X.iloc[i]
            prediction_class, prediction_prob = self.predict_one(entry)
            predictions.append(prediction_class)

        return np.array(predictions)

    def print_feature_posteriors(self):
        """
        For each feature and each of its values, print P(class | feature=value).
        """
        for feature in self.features:
            print(f"\n=== Feature: {feature} ===")
            values = sorted(self.feature_distinct_values[feature])
            for v in values:
                # numerator for each class: P(x=v | C) * P(C)
                numerators = {
                    c: self.per_class_feature_probs[c][feature][v] * self.output_class_probs[c]
                    for c in self.output_classes
                }
                total = sum(numerators.values())
                # normalize to get P(C | x=v)
                posteriors = {c: numerators[c]/total for c in numerators}
                print(f"Value = {v!r}:")
                for c, p in posteriors.items():
                    print(f"  P({c} | {feature}={v!r}) = {p:.4f}")

def load_dataset():
    try:
        return pd.read_csv("mushroom_dataset.csv")
    except FileNotFoundError:
        print("Error: mushroom_dataset.csv not found in the current directory")
        return None

def preprocess_data(df):
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Extract the target variable (class)
    y = data['class'].values
    
    # Drop the target variable from features
    X = data.drop('class', axis=1)
    
    # Label encode all categorical features
    encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        encoders[column] = le
    
    return X, y, encoders

def train_nbc_off_the_shelf(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize and train the model
    nb_model = CategoricalNB()
    nb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = nb_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return nb_model, accuracy, report, X_train, X_test, y_train, y_test


def main():
    # Load data
    print("Loading mushroom dataset...")
    df = load_dataset()
    if df is None:
        print("Could not load dataset. Exiting..")
        return
    
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    
    # Preprocess
    print("Preprocessing data...")
    X, y, encoders = preprocess_data(df)
    
    # Train model
    print("Training Naive Bayes classifier (off-the-shelf)...")
    model, accuracy, report, X_train, X_test, y_train, y_test = train_nbc_off_the_shelf(X, y)
    
    print(f"off-the-shelf model accuracy: {accuracy:.4f}")

    # print("Classification report:")
    # print(report)

    # custom implementation
    nbc = NBC()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Training Naive Bayes classifier (custom implementation)...")
    nbc.train(X_train, y_train)
    nbc.print_feature_posteriors()

    y_pred = nbc.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f"Custom Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()