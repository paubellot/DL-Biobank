"""The genome to be evolved."""

import random
import logging
import hashlib
import copy

from evolve_cnn.train import train_and_score


class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """

    def __init__(self, all_possible_genes=None, geneparam={}, u_ID=0, mom_ID=0, dad_ID=0, gen=0):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Parameters for the genome, includes:
                gene_nb_neurons (list): [64, 128, 256]
                gene_nb_layers (list):  [1, 2, 3, 4]
                gene_activation (list): ['relu', 'elu']
                gene_optimizer (list):  ['rmsprop', 'adam']
                gene_dropout (list): [ 0.   ,  0.075,  0.15 ,  0.225,  0.3  ]
                gene_weight_decay (list): [ 0.   ,  0.075,  0.15 ,  0.225,  0.3  ]
                gene_size: [2,3,5,10]
                gene_batch_norm: [True,False]
                gene_filters: [16, 32, 64, 128]
                gene_size: [2,3,5,10]
                gene_stride: ["equal","one"]
                gene_nb_cnn_layers': [1, 2, 3]
        """
        self.r = -1.0
        self.all_possible_genes = all_possible_genes
        self.geneparam = geneparam  # (dict): represents actual genome parameters
        self.u_ID = u_ID
        self.parents = [mom_ID, dad_ID]
        self.generation = gen

        # hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()

    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        all_possible_genes = {
            'nb_neurons': [16, 32, 64, 128],
            'nb_layers': [1, 2, 3],
            'nb_cnn_layers': [1, 2, 3],
            'batch_norm': [True,False],
            'activation': ['relu', 'elu', 'softplus', 'linear'],
            'optimizer': ['rmsprop', 'nadam'],
            'dropout': [0.,  0.075],
            'filters': [16, 32, 64, 128],
            'size_window': [2,3,5,10],
            'stride' : ["equal","one"],
            'weight_decay': [0., 0.075]
         }
        """
        genh = str(self.geneparam['nb_neurons']) + self.geneparam['activation'] \
               + str(self.geneparam['nb_layers']) + self.geneparam['optimizer'] \
               + str(self.geneparam['dropout']) + str(self.geneparam['weight_decay']) \
               + str(self.geneparam['nb_cnn_layers']) + str(self.geneparam['batch_norm']) \
               + str(self.geneparam['filters']) + str(self.geneparam['size_window']) + self.geneparam['stride']

        self.hash = hashlib.sha256(genh.encode("UTF-8")).hexdigest()

        self.r = -1.0

    def set_genes_random(self):
        """Create a random genome."""
        # print("set_genes_random")
        self.parents = [0, 0]  # very sad - no parents :(

        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])

        self.update_hash()

    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        # Which gene shall we mutate? Choose one of N possible keys/genes.
        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))

        # And then let's mutate one of the genes.
        # Make sure that this actually creates mutation
        current_value = self.geneparam[gene_to_mutate]
        possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])

        possible_choices.remove(current_value)

        self.geneparam[gene_to_mutate] = random.choice(possible_choices)

        self.update_hash()

    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""

        self.generation = generation
        # logging.info("Setting Generation to %d" % self.generation)

    def set_genes_to(self, geneparam, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        IMPROVE
        """
        self.parents = [mom_ID, dad_ID]

        self.geneparam = geneparam

        self.update_hash()

    def train(self, trainingset):
        """Train the genome and record the accuracy.

        Args:
            trainingset (str): Name of dataset to use.

        """
        if self.r == -1.0:
            self.r = train_and_score(self.geneparam, trainingset)

    def print_genome(self):
        """Print out a genome."""
        logging.info(self.geneparam)
        logging.info("R: %.2f%%" % self.r)
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        logging.info(self.geneparam)
        logging.info("R: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (
            self.r , self.u_ID, self.parents[0], self.parents[1], self.generation))
        logging.info("Hash: %s" % self.hash)
