import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

# Convert labels from alphabetic to numeric
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Find the maximum length of any sequence in your data
max_len = max(len(x) for x in data)

# Pad sequences manually with zeros
data_padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in data])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Initialize the RandomForestClassifier
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model using pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)