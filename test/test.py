import torch.multiprocessing as mp
import torch
import time
from mapreduce import Peer
import bittensor as bt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--validator.uid', type = int, default= 0, help='Validator UID')
parser.add_argument('--netuid', type = int, default=10, help='Map Reduce Subnet NetUID')

bt.subtensor.add_args(parser)
bt.wallet.add_args(parser)

config = bt.config(
    parser=parser
)

wallet = bt.wallet(config=config)

# tensor size for testing, set to 10 MB
tensor_size = 10 * 1024 * 1024
bandwidth = tensor_size * 4 # torch.float32 is 4 bytes

def train(rank, peer_count, bandwidth, wallet, validator_uid, network ):
    bt.logging.info(f"🔷 Starting peer with rank {rank}")
    # Initialize Peer instance
    peer = Peer(rank, peer_count, bandwidth, wallet, validator_uid, network)

    # Initialize process group with the fetched configuration
    peer.init_process_group()

    weights = None

    if rank == 1: # if it is the first peer
        weights = torch.rand((tensor_size, 1), dtype=torch.float32)
        # First peer broadcasts the weights
        peer.broadcast(weights)
    else:
        # Other peers receive the weights
        weights = peer.broadcast(weights)
    
    # Should destroy process group after broadcasting
    peer.destroy_process_group()

    # Number of epochs
    epoch = 2

    # Your training loop here
    bt.logging.info(f"Peer {rank} is training...")   
    
    for i in range(epoch):

        bt.logging.success(f"🟢 Epoch: {i}")
        
        # Replace this with actual training code
        time.sleep(5)
        
        # After calculating gradients
        gradients = torch.ones((tensor_size, 1), dtype=torch.float32)
        
        if rank == 1:
            gradients = torch.ones((tensor_size, 1), dtype=torch.float32) * 3
        
        # Initialize process group
        peer.init_process_group()
        
        # All-reducing the gradients (average of gradients)
        gradients = peer.all_reduce(gradients)
        
        # Destroy process group
        peer.destroy_process_group()
    
    bt.logging.success(f"Peer {rank} has finished training.")

bt.logging.info(f"config {config}")

def main():
    peer_count = 3
    processes = []

    # Start two peer processes
    for rank in range(1, peer_count + 1):
        p = mp.Process(target=train, args=(rank, peer_count, tensor_size, wallet, config.validator.uid, config.subtensor.network))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    # mp.set_start_method('spawn')  # This is often necessary in PyTorch multiprocessing
    main()
