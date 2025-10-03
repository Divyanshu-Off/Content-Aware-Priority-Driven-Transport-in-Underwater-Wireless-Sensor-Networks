"""
train_priority_clf.py
Trains a small Decision Tree classifier to detect critical vs non-critical packets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import argparse

def main(args):
    df = pd.read_csv(args.input_csv)
    # We'll do multiclass classification on priority (0..3)
    X = df[["reading","reading_delta","moving_std","battery","depth","pkt_size"]].fillna(0)
    y = df["priority"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Decision rules:\n", export_text(clf, feature_names=list(X.columns)))
    joblib.dump(clf, args.out_model)
    print("Saved model to", args.out_model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="synth_output/dataset_ml.csv")
    parser.add_argument("--out_model", default="priority_clf.pkl")
    args = parser.parse_args()
    main(args)
