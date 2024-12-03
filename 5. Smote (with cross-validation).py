import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

np.random.seed(420)

# Assuming `diabetes_data` is already loaded with the CLASS column
X = diabetes_data.drop(['CLASS'], axis=1)
y = diabetes_data['CLASS']

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns.difference(['Gender'])),
        ('cat', OneHotEncoder(), ['Gender'])
    ])

X = preprocessor.fit_transform(X)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Manual cross-validation for traditional classifiers with SMOTE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

results = {}

for name, clf in classifiers.items():
    accuracies = []
    f1_scores = []
    conf_matrices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE to the training data
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        accuracies.append(acc)
        f1_scores.append(f1)
        conf_matrices.append(cm)

    # Store results
    results[name] = {
        'accuracy': np.mean(accuracies),
        'f1_score': np.mean(f1_scores),
        'conf_matrix': np.mean(conf_matrices, axis=0)
    }

# Manual cross-validation for ANN with SMOTE

def create_ann(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

ann_accuracies = []
ann_f1_scores = []
ann_conf_matrices = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply SMOTE to the training data
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    y_train_nn = to_categorical(y_train_res, num_classes=3)
    y_test_nn = to_categorical(y_test, num_classes=3)

    ann_model = create_ann(X.shape[1])
    ann_model.fit(X_train_res, y_train_nn, epochs=50, batch_size=32, verbose=0)

    _, ann_accuracy = ann_model.evaluate(X_test, y_test_nn, verbose=0)
    y_pred_ann = ann_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_ann, axis=1)

    f1 = f1_score(y_test, y_pred_classes, average='macro')
    cm = confusion_matrix(y_test, y_pred_classes)

    ann_accuracies.append(ann_accuracy)
    ann_f1_scores.append(f1)
    ann_conf_matrices.append(cm)

results['ANN'] = {
    'accuracy': np.mean(ann_accuracies),
    'f1_score': np.mean(ann_f1_scores),
    'conf_matrix': np.mean(ann_conf_matrices, axis=0)
}

# Manual cross-validation for CNN with SMOTE
cnn_accuracies = []
cnn_f1_scores = []
cnn_conf_matrices = []

def create_cnn(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(128, kernel_size=2, activation='relu'))
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply SMOTE to the training data
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Collect predictions from classifiers for CNN input
    train_preds, test_preds = [], []

    for name, clf in classifiers.items():
        clf.fit(X_train_res, y_train_res)
        train_preds.append(clf.predict(X_train_res))
        test_preds.append(clf.predict(X_test))

    # Reshape for CNN
    X_train_cnn = np.array(train_preds).T.reshape((len(train_preds[0]), len(classifiers), 1))
    X_test_cnn = np.array(test_preds).T.reshape((len(test_preds[0]), len(classifiers), 1))

    y_train_nn = to_categorical(y_train_res, num_classes=3)
    y_test_nn = to_categorical(y_test, num_classes=3)

    cnn_model = create_cnn((X_train_cnn.shape[1], 1))
    cnn_model.fit(X_train_cnn, y_train_nn, epochs=50, batch_size=32, verbose=0)

    _, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_nn, verbose=0)
    y_pred_cnn = cnn_model.predict(X_test_cnn)
    y_pred_classes = np.argmax(y_pred_cnn, axis=1)

    f1 = f1_score(y_test, y_pred_classes, average='macro')
    cm = confusion_matrix(y_test, y_pred_classes)

    cnn_accuracies.append(cnn_accuracy)
    cnn_f1_scores.append(f1)
    cnn_conf_matrices.append(cm)

results['CNN'] = {
    'accuracy': np.mean(cnn_accuracies),
    'f1_score': np.mean(cnn_f1_scores),
    'conf_matrix': np.mean(cnn_conf_matrices, axis=0)
}

# Plot all confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for ax, (name, metrics) in zip(axes.flatten(), results.items()):
    conf_matrix = metrics['conf_matrix']
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    annotations = np.array([['{:.1f}%'.format(value) for value in row] for row in conf_matrix_normalized])

    sns.heatmap(conf_matrix_normalized, annot=annotations, fmt='', cmap='Blues', cbar_kws={'format': '%.0f%%'}, ax=ax)
    ax.set_title(f'{name}')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

plt.tight_layout()
plt.show()