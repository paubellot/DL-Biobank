"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import numpy as np
import pandas as pd

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log_biobank_10k.txt'
)



def train_networks(networks, dataset, store_val, store_tst):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset, store_val, store_tst)
        pbar.update(1)
    pbar.close()


def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average accuracy of a population of networks.
    """
    if networks[0].reg:
        vec = np.asarray([])
        for network in networks:
            vec = np.append(vec,network.mse)
        return vec[~np.isnan(vec)].mean()
    else:
        total_accuracy = 0
        for network in networks:
            total_accuracy += network.accuracy

        return total_accuracy / len(networks)


def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()


def generate(generations, population, nn_param_choices, dataset):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    df_val = pd.DataFrame(columns=["nb_layers", "nb_neurons", "activation", "optimizer", "r2"])
    df_tst = pd.DataFrame(columns=["nb_layers", "nb_neurons", "activation", "optimizer", "r2"])

    store_val = pd.io.pytables.HDFStore("val_biobank.h5")
    store_tst = pd.io.pytables.HDFStore("tst_biobank.h5")
    store_val['df'] = df_val
    store_tst['df'] = df_tst
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset, store_val, store_tst)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        if networks[0].reg:
            logging.info("Generation average: %.2f%%" % (average_accuracy))
        else:
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.mse)
    store_val.close()
    store_tst.close()
    # Print out the top 5 networks.
    print_networks(networks[:5])


def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.
    dataset = '10k'

    nn_param_choices = {
        'nb_neurons': [8, 16, 32, 64, 128, 256, 512],
        #'nb_neurons': [ 2, 3, 4, 5, 10, 13, 26, 52],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu','sigmoid','selu'],
        'optimizer': ['rmsprop'],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main()