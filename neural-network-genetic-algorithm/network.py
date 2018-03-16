"""Class that represents the network to be evolved."""
import random
import logging
import numpy as np
from train import train_and_score


class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """
    mse = 10000.
    accuracy = 0.
    reg = False
    model = []
    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.mse = 10000.
        self.accuracy = 0.
        self.reg = False
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset, store_val, store_tst):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if dataset == "mg" or dataset == "boston" or dataset == "10k":
            self.reg = True
            if self.mse == 10000.:
                self.mse = train_and_score(self, dataset, store_val, store_tst)

        else:
            if self.accuracy == 0.:
                self.accuracy = train_and_score(self, dataset)


    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        if self.reg:
            logging.info("Network mse: %.2f" % (self.mse))
        else:
            logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
