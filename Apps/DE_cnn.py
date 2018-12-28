import argparse
import logging
from GA.evolve_cnn.evolver import Evolver


class DS:
    def __init__(self, trait="height", k=10000, unif=False):
        self.trait = trait
        self.k = k
        self.unif = unif


def train_genomes(genomes, dataset):
    """Train each genome.
    Args:
        genomes (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating
    """
    logging.info("***train_networks(networks, dataset)***")

    for genome in genomes:
        genome.train(dataset)


def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.
    Args:
        genomes (list): List of networks/genomes
    Returns:
        float: The average accuracy of a population of networks/genomes.
    """
    total_accuracy = 0

    for genome in genomes:
        total_accuracy += genome.r

    return total_accuracy / len(genomes)


def generate(generations, population, all_possible_genes, dataset):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")

    evolver = Evolver(all_possible_genes)

    genomes = evolver.create_population(population)

    # Evolve the generation.
    for i in range(generations):

        logging.info("***Now in generation %d of %d***" % (i + 1, generations))

        print_genomes(genomes)

        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(genomes)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)  # -----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.r, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:5])


def print_genomes(genomes):
    """Print a list of genomes.
    Args:
        genomes (list): The population of networks/genomes
    """
    logging.info('-'*80)

    for genome in genomes:
        genome.print_genome()


def main(trait, k, population=30, generations=8):
    """
    Find a CNN with GA
    :param trait: Str; name of the trait to fit a MLP
    :param k: Int, number of SNPs to use
    :param population: Number of networks/genomes in each generation.
    :param generations: Int; number of generations
    :return: Nothing
    """


    # we only need to train the new ones....
    # generations: Number of times to evolve the population.
    all_possible_genes = {
        'nb_neurons': [16, 32, 64, 128],
        'nb_layers': [1, 2, 3],
        'nb_cnn_layers': [1, 2, 3],
        'batch_norm': [True, False],
        'activation': ['relu', 'elu', 'softplus', 'linear'],
        'optimizer': ['rmsprop', 'nadam'],
        'dropout': [0., 0.075],
        'filters': [16, 32, 64, 128],
        'size_window': [2, 3, 5, 10],
        'stride': ["equal", "one"],
        'weight_decay': [0., 0.075]
    }
    ds = DS(trait, k)

    print("***Dataset:", ds.trait, str(ds.k))

    print("***Evolving for %d generations with population size = %d***" % (generations, population))
    generate(generations, population, all_possible_genes, ds)


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--trait", help="trait to run optimization", default="height")
        parser.add_argument("-k", "--num_snps", help="number of SNPs", default=10000, type=int)

        args = parser.parse_args()
        filename = "log_{0}_{1}k_cnn.txt".format(args.trait, int(args.num_snps / 1000))

        # Setup logging.
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO,
            filename=filename
        )
        main(args.trait, args.num_snps)
