import numpy as np
import pandas as pd
import scipy.optimize as opt

# 1. Load the dataset
data = pd.read_csv('your_dataset.csv')

# 2. Feature extraction (example: path length, number of links)
def extract_features(path):
    nodes = list(map(int, path.split('_')))
    num_links = len(nodes) - 1
    path_length = data[data['path'] == path]['len'].values[0]
    time_step = data[data['path'] == path]['time_step'].values[0]
    # Add other features if available
    return path_length, num_links, time_step

data['features'] = data['path'].apply(extract_features)

# 3. Define utility function
def utility_function(beta, features):
    return np.dot(features, beta)

# 4. Calculate choice probabilities
def choice_probabilities(beta, features):
    utilities = np.dot(features, beta)
    exp_utilities = np.exp(utilities)
    return exp_utilities / np.sum(exp_utilities, axis=0)

# 5. Define log-likelihood function
def log_likelihood(beta, features, chosen_paths):
    probabilities = choice_probabilities(beta, features)
    chosen_log_probs = np.log(probabilities[np.arange(len(chosen_paths)), chosen_paths])
    return -np.sum(chosen_log_probs)  # we want to minimize the negative log-likelihood

# 6. Prepare data for model
features = np.array([extract_features(path) for path in data['path']])
chosen_paths = np.arange(len(data))  # Assuming each row represents the chosen path

# 7. Initial guess for beta coefficients
initial_beta = np.zeros(features.shape[1])

# 8. Estimate parameters using optimization
result = opt.minimize(log_likelihood, initial_beta, args=(features, chosen_paths))

# 9. Extract estimated coefficients
estimated_beta = result.x
print("Estimated Beta Coefficients:", estimated_beta)

# 10. Model evaluation
# Calculate predicted probabilities
predicted_probabilities = choice_probabilities(estimated_beta, features)

# Compare predicted probabilities with observed data
# This step could involve calculating metrics like accuracy, log-likelihood, etc.
