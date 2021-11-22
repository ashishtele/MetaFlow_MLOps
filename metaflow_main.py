from metaflow import FlowSpec, step, parameter
import os
import pandas as pd
import numpy as np

class Pipeline(FlowSpec):
    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        TODO: Load data from file
        """

        self.next(self.preprocess)

    @step
    def split_data(self):
        
        """
        TODO: Split data into training and test sets
        """
        self.next(self.train)

    @step
    def train(self):
        """
        TODO: Train model
        """
        self.next(self.predict)

    @step
    def predict(self):
        """
        TODO: Predict on test set
        """
        self.next(self.end)

    @step
    def end(self):
        print("End of Pipeline!")

if __name__ == '__main__':
    Pipeline()