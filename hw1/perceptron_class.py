import pandas as pd
import numpy as np

# Set up perceptron algorithm to model randomised target functions. track performance in the form of the number of
# iterations for the model to converge on the target function, and the likelihood of mis-classifying a random point in
# the target space.


class Perceptron:

    def __init__(self, target, sample=100):
        self.target = target
        self.dim = len(target)
        self.training_sample_size = sample
        self.testing_sample_size = sample // 2
        self.weights = np.zeros(len(target) + 1)
        self.training_sample = []
        self.testing_sample = []
        self.training_sample_model = np.zeros(self.training_sample_size)
        self.testing_sample_model = np.zeros(self.testing_sample_size)
        self.iterations = 0

    def create_training_sample(self):
        training_inputs = np.random.random((self.training_sample_size, self.dim)) * 2 - 1  # random takes input
        # sample size & dimensions. Final element in each array is the dependent variable, with all previous elements
        # being the independent inputs.
        # For each input point in the 'dim'-dimensional input space:
        for ipt in training_inputs:
            # Calculate the value of the target linear function.
            target_value = 0
            ipt = np.insert(ipt, 0, 1)
            for i in range(self.dim):
                # Multiply all independent variable inputs in input point by their relevant coefficients in the
                # target. Max index of dim - 1 excludes the dependent value in input point.
                target_value += ipt[i] * self.target[i]
            # If dependent value of input point is > target function value, then 1; if <, then -1; if =, then 0.
            if ipt[self.dim] > target_value:
                func_value = 1
            elif ipt[self.dim] < target_value:
                func_value = -1
            else:
                func_value = 0
            # Append each training point to training_sample array along with their function value.
            self.training_sample.append((ipt, func_value))
        return

    def create_testing_sample(self):
        testing_inputs = np.random.random((self.testing_sample_size, self.dim)) * 2 - 1  # random takes input
        # sample size & dimensions. Final element in each array is the dependent variable, with all previous elements
        # being the independent inputs.
        # For each input point in the 'dim'-dimensional input space:
        for ipt in testing_inputs:
            # Calculate the value of the target linear function.
            target_value = 0
            ipt = np.insert(ipt, 0, 1)
            for i in range(self.dim):
                # Multiply all independent variable inputs in input point by their relevant coefficients in the
                # target. Max index of dim - 1 excludes the dependent value in input point.
                target_value += ipt[i] * self.target[i]
            # Add target intercept.
            target_value += self.target[0]
            # If dependent value of input point is > target function value, then 1; if <, then -1; if =, then 0.
            if ipt[self.dim] > target_value:
                func_value = 1
            elif ipt[self.dim] < target_value:
                func_value = -1
            else:
                func_value = 0
            # Append each testing point to testing_sample array along with their function value.
            self.testing_sample.append((ipt, func_value))
        return

    def iterate(self):
        # get function results for training set.
        training_sample_results = pd.array([point[1] for point in self.training_sample])

        # while current iteration function values don't match the target function values for the training set.
        # loop while the modelled results for the training sample don't equal the true results:
        while not (self.training_sample_model == training_sample_results).all():
            # Increase iteration counter.
            self.iterations += 1

            # find first sample point index which doesn't match.
            change_index = 0
            while self.training_sample_model[change_index] == training_sample_results[change_index]:
                change_index += 1

            # update weight vector via perceptron algorithm.
            self.weights[0] += training_sample_results[change_index]
            for i in range(self.dim):
                self.weights[i + 1] += (training_sample_results[change_index]
                                        * self.training_sample[change_index][0][i + 1])

            # iterate the current function values
            training_sample_model_holder = []
            for i in range(self.training_sample_size):
                training_sample_model_holder.append(0)
                # for j in range(self.dim - 1):
                #     training_sample_model_holder[i] += self.training_sample[i][0][j] * self.weights[j + 1]
                # dot product of weights and training point.
                training_sample_model_holder[i] += np.dot(self.weights, self.training_sample[i][0])

            self.training_sample_model = [1 if self.weights[0] + training_sample_model_holder[i] > 0
                                          else -1 if self.weights[0] + training_sample_model_holder[i] < 0
                                          else 0 for i in range(self.training_sample_size)]

            pass
        return

    pass


if __name__ == '__main__':

    test = Perceptron([0.5, 0.5, 0.1, 0.25], sample=10)
    print(test.dim)
    print(test.training_sample_size)
    print(test.testing_sample_size)
    print(test.training_sample)
    print(test.weights)

    test.create_training_sample()
    print(test.training_sample)
    # print(test.training_sample[0][0])

    test.create_testing_sample()
    print(test.testing_sample)

    print()

    print(test.iterations)
    print(test.weights)

    test.iterate()

    print(test.iterations)
    print(test.weights)
