"""Main script from which are executed the features of the project."""
import argparse

import src.benchmark as benchmark


parser = argparse.ArgumentParser(description='MVA RP Data Challenge 2021')
parser.add_argument('script', type=str, help='The subscript to run.')
parser.add_argument('action', type=str, help='The action to run.')
parser.add_argument('--ds', type=str, default='btc', help='The dataset.')
parser.add_argument('--w', type=int, default=24, help='Windows width.')
parser.add_argument('--s', type=int, default=12, help='Windows stride.')

args = parser.parse_args()

# Map custom command to script
mapping = {
}

locals()[mapping.get(args.script, args.script)].run(args)
