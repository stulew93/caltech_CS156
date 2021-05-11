import pandas as pd
import numpy as np

# Set up perceptron algorithm to model randomised target functions. track performance in the form of the number of
# iterations for the model to converge on the target function, and the likelihood of misclassifying a random point in
# the target space. 

# Specify desired training sample size. test sample size will be half the size.
training_sample_size = 100
testing_sample_size = training_sample_size // 2

# Specify number of repeats.
repeats = 1000

# Create list to track number of iterations needed per run, and number of misclassified testing points.
performance_tracker = []

# Iterate perceptron algorithm 1000 times.
for i in range(repeats):
    print("Repeat no. {}".format(i + 1))

    # Create random target function on [-1, 1] x [-1, 1].
    # Slope between -2 and 2.
    slope = np.random.rand() * 2
    if np.random.rand() < 0.5:
        slope *= -1

    # Intercept between -1 and 1.
    intercept = np.random.rand()
    if np.random.rand() < 0.5:
        intercept *= -1

    # Create training sample to train model on.
    training_sample = np.random.random((training_sample_size, 2)) * 2 - 1  # random takes input sample size & dimensions

    # Get the correct target function values for the training set.
    training_sample_y = [1 if training_sample[i][1] > intercept + slope * training_sample[i][0]
                         else -1 if training_sample[i][1] < intercept + slope * training_sample[i][0]
                         else 0 for i in range(training_sample_size)]

    # Create testing sample to test model on. Displayed as a grid.
    testing_sample = np.random.random((testing_sample_size, 2)) * 2 - 1  # random takes input sample size & dimensions

    # Get the correct target function values for the testing set.
    testing_sample_y = [1 if testing_sample[i][1] > intercept + slope * testing_sample[i][0]
                        else -1 if testing_sample[i][1] < intercept + slope * testing_sample[i][0]
                        else 0 for i in range(testing_sample_size)]

    # weights initialised in 2-d space, with a bias element.
    w = [0, 0, 0]

    # initialise current y values calculated using x1 and x2 values.
    training_sample_y_model = [0 for i in range(training_sample_size)]

    # dataframe to track output iterations.
    y_iters = pd.DataFrame([training_sample_y], ['initial'])

    # dataframe to track weight iterations.
    w_iters = pd.DataFrame([w], ['initial'])

    # Intialise iteration counter.
    iters = 0

    # while current iteration function values don't match the target function values for the training set
    while training_sample_y_model != training_sample_y:
        # Increase iteration counter.
        iters += 1
        # find first sample point index which doesn't match.
        change_index = 0
        while training_sample_y_model[change_index] == training_sample_y[change_index]:
            change_index += 1
        # update weight vector via perceptron algorithm.
        w[0] += training_sample_y[change_index]
        w[1] += training_sample_y[change_index] * training_sample[change_index][0]
        w[2] += training_sample_y[change_index] * training_sample[change_index][1]
        # iterate the current function values
        training_sample_y_model = [1 if w[0] + w[1] * training_sample[i][0] + w[2] * training_sample[i][1] > 0
                                   else -1 if w[0] + w[1] * training_sample[i][0] + w[2] * training_sample[i][1] < 0
                                   else 0 for i in range(training_sample_size)]

    # Get model function values for testing set.
    testing_sample_y_model = [1 if w[0] + w[1] * testing_sample[i][0] + w[2] * testing_sample[i][1] > 0
                              else -1 if w[0] + w[1] * testing_sample[i][0] + w[2] * testing_sample[i][1] < 0
                              else 0 for i in range(testing_sample_size)]

    # Count number of points in testing sample which don't have the correct function output using the model.
    misclassified = [1 if testing_sample_y[i] != testing_sample_y_model[i] else 0 for i in range(testing_sample_size)]
    num_misclassified = sum(misclassified)

    # Save number of iterations and number of misclassified points in performance_tracker list.
    performance_tracker.append((iters, num_misclassified))

print(performance_tracker)
df = pd.DataFrame(performance_tracker)
df.columns = ['Iterations', 'Misclassified']
print(df)

# Probability of mis-classification of a given point:
# For each repeat, we have testing_sample_size points. Num_misclassified / testing_sample_size gives chance for given
# repeat. Average this quantity over all repeats to give overall average.
misclassified_chance_per_repeat = df['Misclassified'] / testing_sample_size
print(misclassified_chance_per_repeat)
misclassified_chance_avg = misclassified_chance_per_repeat.mean()

print("Avg iterations required for convergence: {}".format(df["Iterations"].mean()))
print("Avg chance of mis-classification: {}".format(misclassified_chance_avg))
