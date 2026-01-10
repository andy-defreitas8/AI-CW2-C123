# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


# setting a random seed 
random_number = 42
np.random.seed(random_number)

# loading the data
def load_data(filepath):
    
    # dataframe or the dataset
    df = pd.read_csv(filepath)
    return df

# exploring and visualizing the class distribution
def explore_data(df, label_column):
    
    # class distribution
    class_counts = df[label_column].value_counts().sort_index()
    print(f"\nTotal instances: {len(df)}")
    print(f"Number of classes: {len(class_counts)}")
    
    # visualization of the class distribution
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind = 'bar', edgecolor = 'black')
    plt.title('Class Distribution in the ASL Dataset', fontsize = 12, fontweight='bold')
    plt.xlabel('ASL alphabets ', fontsize = 12, fontstyle = 'italic')
    plt.ylabel('Number of Instances', fontsize = 12, fontstyle = 'italic')
    plt.xticks(rotation=0)
    
    # Add value labels on top of each bar
    for i, v in enumerate(class_counts):
        plt.text(i, v, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Class_distribution.png', dpi=300)
    plt.show()
    return class_counts


# splitting the data in training and testing; testing data size is default; 0.2 = 20%
# X = features
# y = labels

def split_data(df, label_column, test_size=0.2):
    
    # separating features and labels
    X = df.drop(columns=[label_column]) # features
    y = df[label_column] # labels
    
    
    # removing the image_name columns in the excel sheet
    non_feature_cols = ['image_name']
    cols_removed = []
    for col in non_feature_cols:
        if col in X.columns:
            X = X.drop(columns=[col])
            cols_removed.append(col)
    
    return train_test_split(
        X, y, test_size = test_size, random_state = random_number, stratify = y
    )

# tuning the decision tree parameters using 5-fold cross validation. 
# X_train = training features
# y_train = training labels
# n_folds = number of folds for cross validation (default is 5)

def tune_hyperparameters(X_train, y_train, n_folds=5):
    
    # maximum depth of the tree (controls how complex the tree can be)
    # none = no limits
    # smaller numbers = shallower tree (less likely to overfit)
    max_depth_values = [None, 5, 10, 15, 20, 25]
    
    # minimum samples required to split an internal node
    # Smaller numbers = tree can split more easily (more complex)
    # Larger numbers = tree is more constrained (simpler, less overfitting)
    min_samples_split_values = [2, 5, 10, 20, 50]
    
    
    # list to store cross-validation results for each combination.
    results = []
    
    # testing all the combinations
    
    # grid search; nested loops to iterate through all pairs of hyperparameters
    for depth in max_depth_values:
        for sam_split in min_samples_split_values:
            
            #creating a classifier with current hyperparamters
            clf = DecisionTreeClassifier(
                max_depth = depth,
                min_samples_split = sam_split,
                random_state = random_number
            )
            
            # performing 5-fold cross validation
            scores = cross_val_score(clf, X_train, y_train, cv=n_folds)
            
            # storing the results
            # cv_mean_accuracy = average accuracy across folds 
            # cv_std_accuracy = standard deviation (measure of consistency)
            results.append({
                'max_depth': 'None' if depth is None else depth,
                'min_samples_split': sam_split,
                'cv_mean_accuracy': scores.mean(),
                'cv_std_accuracy': scores.std()
            })
     
    # makes a nice table of all combinations and their CV scores
    results_df = pd.DataFrame(results)
    
    # finding the best parameters
    # idxmax() = returns the row index with highest mean CV accuracy
    
    best_idx = results_df['cv_mean_accuracy'].idxmax()
    
    # picking the best combination of hyperparameters
    best_params = {
        'max_depth': None if results_df.loc[best_idx, 'max_depth'] == 'None'
        else results_df.loc[best_idx, 'max_depth'],
        'min_samples_split': results_df.loc[best_idx, 'min_samples_split']
    }
    return results_df, best_params



# Usage


def train_and_evaluate_best_model(X_train, X_test, y_train, y_test, best_params):
    
    # creating model with best hyperparameters
    model = DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=random_number
    )
    
    # training the model
    model.fit(X_train, y_train)
    
    # making prediction on training set; checking for overfitting
    y_train_pred = model.predict(X_train)
    
    # making prediction on testing set; checking generalization performance
    y_test_pred = model.predict(X_test)
    
    # printing the detailed test set performance
    print ("printing the detailed test set performance")
    print(classification_report(y_test, y_test_pred))
    return model, y_train_pred, y_test_pred

# generating confusion matrices of training and testing set

def plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, classes):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(confusion_matrix(y_train, y_train_pred, labels=classes),
                annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=classes, yticklabels=classes )
    axes[0].set_title(f'Confusion Matrix - Training Set\nAccuracy: {accuracy_score(y_train, y_train_pred) * 100:.4f}', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Model's Prediction", fontsize=12)
    axes[0].set_ylabel("Actual Class", fontsize=12)

    sns.heatmap(confusion_matrix(y_test, y_test_pred, labels=classes),
                annot=True, fmt='d', cmap='Purples', ax=axes[1], xticklabels=classes, yticklabels=classes)
    axes[1].set_title(f'Confusion Matrix - Test Set\nAccuracy: {accuracy_score(y_test, y_test_pred) * 100:.4f}', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Model's Prediction", fontsize=12)
    axes[1].set_ylabel("Actual Class", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('Confusion_matrices.png', dpi=300)
    plt.show()
    
# y_test = the actual class of test set
# y_test_pred = the predicted class from the model

def visualize_classification_report(y_test, y_test_pred, classes):
    
    report = classification_report(y_test, y_test_pred, target_names=classes, output_dict=True)
    precision = [report[i]['precision'] * 100 for i in classes]
    recall = [report[j]['recall'] * 100 for j in classes]
    f1 = [report[k]['f1-score'] * 100 for k in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='royalblue')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='lightseagreen')
    
    # adding value labels on top of each bar
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 120)  
    
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    
    # average scores for legend
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    # custom legend with percentages
    ax.legend([f'Precision ({avg_precision:.1f}%)', 
               f'Recall ({avg_recall:.1f}%)', 
               f'F1-Score ({avg_f1:.1f}%)'])
    
    plt.tight_layout()
    plt.savefig('Classification_report.png', dpi=300)
    plt.show()
    
# creating a summary table of the model performance
def create_performance_summary(results_df, best_params, train_acc, test_acc):
    
    best_cv = results_df['cv_mean_accuracy'].max()
    print ("creating a summary table of the model performance")
    #print(best_cv, train_acc, test_acc, best_params)
    
    #print ("Best Cross Validation Accuracy (5 -fold): ", best_cv)
    print(f"Best Cross Validation Accuracy (5 -fold): {best_cv * 100:.2f}%")

    #print ("Best Model Training Accuracy:", train_acc)
    print(f"Best Model Training Accuracy:: {train_acc * 100:.2f}%")
    
    #print ("Best Model Test Accuracy:", test_acc)
    print(f"Best Model Test Accuracy:: {test_acc * 100:.2f}%")
    
    print ("Best max_depth:", best_params['max_depth'])
    
    print ("Best min_samples_split: ", best_params['min_samples_split'])

# main execution

if __name__ == "__main__":
    
    #loading and exloring the data
    filepath = "combined_landmarks_clean.csv"
    label_column = "label"

    df = load_data(filepath)
    explore_data(df, label_column)
    
    # splitting the data to training and testing data
    X_train, X_test, y_train, y_test = split_data(df, label_column)
    
    # hyperparamater tuning
    results_df, best_params = tune_hyperparameters(X_train, y_train)
    
    # training the best model and evaluating
    model, y_train_pred, y_test_pred = train_and_evaluate_best_model(
        X_train, X_test, y_train, y_test, best_params
    )
    
    # plotting confusion matrices and classification report
    classes = sorted(df[label_column].unique())
    plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, classes)
    visualize_classification_report(y_test, y_test_pred, classes)
    
    # creating performance summary
    create_performance_summary(
        results_df,
        best_params,
        accuracy_score(y_train, y_train_pred),
        accuracy_score(y_test, y_test_pred)
    )
