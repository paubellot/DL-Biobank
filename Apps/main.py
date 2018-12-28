import argparse
import logging
from Apps import DE_cnn
from Apps import DE_mlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", help="Trait to run optimization", default="height")
    parser.add_argument("-k", "--num_snps", help="Number of SNPs", default=10000, type=int)
    parser.add_argument("--unif", help="Use uniformly spaced spns", action='store_true')
    parser.add_argument("--population", help="Population GA",default=30, type=int)
    parser.add_argument("--generations",help="Number of generations GA",default= 8,type=int)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mlp', action='store_true')
    group.add_argument('--cnn', action='store_true')

    args = parser.parse_args()
    
    if args.mlp:
        filename = "log_" + args.trait + "_" + str(args.num_snps / 1000) + "k.txt"
        # Setup logging.
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO,
            filename=filename
        )
        DE_mlp.main(trait=args.trait, k=args.num_snps, population=args.population, generations = args.generations)
    if args.cnn:
        filename = "log_" + args.trait + "_" + str(args.num_snps / 1000) + "k_cnn.txt"
        # Setup logging.
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO,
            filename=filename
        )
        DE_cnn.main(trait=args.trait, k=args.num_snps, population=args.population, generations = args.genrations)

