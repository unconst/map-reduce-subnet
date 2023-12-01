import torch.multiprocessing as mp
import time
from mapreduce.peer import Peer
import mapreduce.utils as utils
import bittensor as bt
from argparse import ArgumentParser
import os

# Maximum size for benchmarking, set to 1 GB
benchmark_max_size = 1 * 1024 * 1024 * 1024 

# Setting up the argument parser
parser = ArgumentParser()

parser.add_argument('--validator.uid', type = int, help='Validator UID')
parser.add_argument('--netuid', type = int, default = 10, help = "The chain subnet uid." )

bt.subtensor.add_args(parser)
bt.logging.add_args(parser)
bt.wallet.add_args(parser)
bt.axon.add_args(parser)

config = bt.config(
    parser=parser
)

wallet = bt.wallet(config=config)
dendrite = bt.dendrite(wallet=wallet) 
validator_uid = config.validator.uid

config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'benchmark',
        )
    )
# Ensure the directory for logging exists, else create one.
if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)

bt.logging(config=config, logging_dir=config.full_path)

bt.logging.info(f"Configurations: {config}")
"""
Performs benchmarking operations for the given validator.

Args:
    wallet (bt.Wallet): Bittensor wallet instance.
    validator_uid (int): Validator's unique identifier.
    netuid (int): Network unique identifier.
    network (str): Network information.
"""
def benchmark(wallet, validator_uid, netuid, network ):
    bt.logging.info("")
    bt.logging.info(f"Starting benchmarking bot")
    
    # Initialize Peer instance
    peer = Peer(1, 1, 0, wallet, validator_uid, netuid, network, benchmark_max_size = benchmark_max_size)

    # Initialize process group with the fetched configuration
    if not peer.benchmark():
        # Not able to benchmark, wait a bit
        time.sleep(2)
        return
    peer.destroy_process_group()
    bt.logging.success("Benchmarking completed")

"""
The main function to continuously run the benchmarking process.
"""
def main():
    while True: 
        # Start the benchmarking process
        p = mp.Process(target=benchmark, args=(wallet, validator_uid, config.netuid, config.subtensor.network))
        p.start()
        # Wait for the process to complete, with a specified timeout
        p.join(timeout=60)  # Set your desired timeout in seconds
        
        # Check if the process is still alive after the timeout
        if p.is_alive():
            # If the process is still running after the timeout, terminate it
            bt.logging.warning("Benchmark process exceeded timeout, terminating...")
            p.terminate()
            # Wait a bit for the process to terminate
            p.join()
        
        # Sleep for a short duration before starting the next process
        time.sleep(1)

if __name__ == '__main__':
    # Check if there is enough free memory to run the benchmark
    if utils.get_available_memory() < benchmark_max_size * 2:
        bt.logging.error("🔴 Not enough memory to run benchmark")
        exit(1)
    main()