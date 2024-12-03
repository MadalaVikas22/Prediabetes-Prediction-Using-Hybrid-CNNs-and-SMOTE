np.random.seed(420)

X = diabetes_data.drop(['CLASS'], axis=1)
y = diabetes_data['CLASS']

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns.difference(['Gender'])),
        ('cat', OneHotEncoder(), ['Gender'])
    ])

X = preprocessor.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter = 10000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

results = {}

for name, clf in classifiers.items():
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'conf_matrix': cm
    }

# ANN Model
def create_ann(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

y_train_nn = to_categorical(y_train_res, num_classes=3)
y_test_nn = to_categorical(y_test, num_classes=3)

ann_model = create_ann(X_train_res.shape[1])
ann_model.fit(X_train_res, y_train_nn, epochs=50, batch_size=32, verbose=0)

_, ann_accuracy = ann_model.evaluate(X_test, y_test_nn, verbose=0)
y_pred_ann = ann_model.predict(X_test)
y_pred_classes_ann = np.argmax(y_pred_ann, axis=1)

ann_f1 = f1_score(y_test, y_pred_classes_ann, average='macro')
ann_cm = confusion_matrix(y_test, y_pred_classes_ann)

results['ANN'] = {
    'accuracy': ann_accuracy,
    'f1_score': ann_f1,
    'conf_matrix': ann_cm
}

# CNN Model
def create_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(128, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Collect predictions from classifiers for CNN input
train_preds, test_preds = [], []

for name, clf in classifiers.items():
    clf.fit(X_train_res, y_train_res)
    train_preds.append(clf.predict(X_train_res))
    test_preds.append(clf.predict(X_test))

# Reshape for CNN
X_train_cnn = np.array(train_preds).T.reshape((len(train_preds[0]), len(classifiers), 1))
X_test_cnn = np.array(test_preds).T.reshape((len(test_preds[0]), len(classifiers), 1))

cnn_model = create_cnn((X_train_cnn.shape[1], 1))
cnn_model.fit(X_train_cnn, y_train_nn, epochs=50, batch_size=32, verbose=0)

_, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_nn, verbose=0)
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_classes_cnn = np.argmax(y_pred_cnn, axis=1)

cnn_f1 = f1_score(y_test, y_pred_classes_cnn, average='macro')
cnn_cm = confusion_matrix(y_test, y_pred_classes_cnn)

results['CNN'] = {
    'accuracy': cnn_accuracy,
    'f1_score': cnn_f1,
    'conf_matrix': cnn_cm
}

# Plot confusion matrices
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
