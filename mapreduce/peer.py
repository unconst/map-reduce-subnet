# peerlib.py

import os
import torch.distributed as dist
import torch
from math import ceil
import bittensor as bt
import mapreduce
from mapreduce.utils import get_my_version, set_gloo_socket_ifname, chunk_with_padding, human_readable_size, merge_chunks
from datetime import timedelta
import gc

class Peer:
    def __init__(self, rank, peer_count, bandwidth, wallet: bt.wallet, validator_uid: int = -1, network = 'finney', port_range = "9000:9010", benchmark_max_size = 0):
        
        netuid = 10
        assert rank > 0, "Rank should not be zero"
        
        metagraph = bt.metagraph(netuid, network=network)
        metagraph.sync()
        self.axons = metagraph.axons

        if validator_uid >=  0:
            self.validator_uid = validator_uid
        else:
            self.validator_uid = 0
            
        self.hotkey = wallet.hotkey
        self.rank = rank
        self.peer_count = peer_count
        self.bandwidth = bandwidth
        self.world_size = None
        self.validator_addr = None
        self.validator_port = None
        self.wallet = wallet
        self.dendrite: bt.dendrite = bt.dendrite(wallet=self.wallet)
        self.port_range = port_range
        self.benchmark_max_size = int(benchmark_max_size)
        
        # Set port range for gloo
        os.environ['GLOO_PORT_RANGE'] = port_range
        
        if benchmark_max_size:
            data_length = int(benchmark_max_size / 4)
            bt.logging.info(f"Max benchmark size: {human_readable_size(benchmark_max_size)}")
            self.temp_tensor = torch.rand(data_length, 1)
        
    def fetch_config_from_validator(self):

        query = mapreduce.protocol.ConnectMaster(
            version=get_my_version(),
            job=mapreduce.protocol.Job(
                rank=self.rank,
                peer_count=self.peer_count,
                bandwidth=self.bandwidth
            ),
        )
        
        bt.logging.info(f"External IP: {self.dendrite.external_ip}")
        set_gloo_socket_ifname(self.dendrite.external_ip)

        response: mapreduce.protocol.ConnectMaster = self.dendrite.query(
            axons = self.axons[self.validator_uid],
            synapse = query,
            timeout = 30
        )
        bt.logging.trace("🔵 Response from validator")
        bt.logging.trace(response)
        
        if response.job.status == "error":
            raise Exception(response.job.reason)
        if response.job.status is None:
            raise Exception("Unable to connect to validator")
        
        self.world_size = response.job.world_size
        self.master_addr = response.job.master_addr
        self.master_port = response.job.master_port

    def init_process_group(self):
        self.fetch_config_from_validator()
        
        bt.logging.info(f"Joining group tcp://{self.master_addr}:{self.master_port} rank: {self.rank}")
        dist.init_process_group(
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            backend='gloo',
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=10)
        )
        bt.logging.info('Initialized process group')
        self.init_groups()

    def init_groups(self):
        self.peers = dist.new_group(ranks=list(range(1, self.peer_count + 1)))
        self.miners = dist.new_group(ranks=list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1)))
        self.peer_miners = [dist.new_group(ranks=[peer_rank] + list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1))) for peer_rank in range(1, self.peer_count + 1)]
        self.miner_peers = [dist.new_group(ranks=[miner_rank] + list(range(1, self.peer_count + 1))) for miner_rank in range(self.peer_count + 1, self.peer_count + self.peer_count + 1)]
        self.validator_rank1 = dist.new_group(ranks=(0, 1))
        bt.logging.info('Groups created')

    def destroy_process_group(self):
        if self.rank == 1:
            bt.logging.info("🔔 Action: Exit")
            dist.broadcast_object_list([{ 'type': 'exit' }], src=1, group = self.validator_rank1)
        actions = [{}] 
        dist.broadcast_object_list(actions, src=0)
        action = actions[0]
        if action['type'] == 'exit':
            bt.logging.info(f'🟢 Received \033[91mexit\033[0m signal')
        dist.destroy_process_group()

    def scatter_to_miners(self, tensor):
        # Inform validator to scatter tensor to all miners
        chunks = chunk_with_padding(tensor, self.peer_count)
        temp = torch.empty_like(chunks[0])
        # concat temp + chunks
        chunks = [temp] + chunks
        # calculate tensor size in bytes
        tensor_size = tensor.element_size() * tensor.nelement()

        bt.logging.info(f"⌛ Broadcasting \033[92m{human_readable_size(tensor_size)}\033[0m")
        dist.scatter(temp, chunks, src=self.rank, group=self.peer_miners[self.rank-1])
        del temp
        del chunks
        gc.collect()
        bt.logging.info("✅ Broadcast finished")

    # Gather data from miners
    def gather_from_miners(self, shape, dtype, chunk_shape):
        gather_list = [torch.empty(chunk_shape, dtype=dtype) for _ in range(self.peer_count+1)]
        bt.logging.info("⌛ Gathering ...")
        gc.collect()
        dist.gather(gather_list[0], gather_list, dst=self.rank, group=self.peer_miners[self.rank-1])
        bt.logging.info("✅ Gather finished")
        merged_chunks = merge_chunks(gather_list[1:], shape[0])
        del gather_list
        gc.collect()
        return merged_chunks

    # Each peer gathers chunks from all miners
    def reduce_from_peers(self, tensor):
        bt.logging.info('⌛ Waiting for Reducing ...')
        chunks = chunk_with_padding(tensor, self.peer_count)
        handles = []
        for i in range(self.peer_count):
            miner_rank = i + self.peer_count + 1
            handles.append(dist.reduce(chunks[i], dst=miner_rank, group=self.miner_peers[i], async_op=True))
        for i in range(self.peer_count):
            handles[i].wait()
            del chunks[0]
        del tensor
        del chunks
        gc.collect()
        bt.logging.info('✅ Reduced to all miners')

    # Broadcasting tensor to all peers
    def broadcast(self, tensor: torch.Tensor):
        if self.rank == 1:
            bt.logging.info('🔔 Send Action: broadcast')
            chunk_shape = (ceil(tensor.size(0) / self.peer_count), ) + tensor.shape[1:]
            dist.broadcast_object_list([{
                'type': 'broadcast', 
                'src': 1, 
                'shape': tuple(tensor.shape),
                'chunk_shape': chunk_shape,
                'dtype': tensor.dtype,
                }], src=1, group = self.validator_rank1)
            actions = [{}] 
            dist.broadcast_object_list(actions, src=0)
            bt.logging.info(f"🔔 Action: {actions[0]}")
            self.scatter_to_miners(tensor)
        else:
            actions = [{}] 
            dist.broadcast_object_list(actions, src=0)
            action = actions[0]
            bt.logging.info(f"🔔 Action: {action}")
            return self.gather_from_miners(action['shape'], action['dtype'], action['chunk_shape'])
        
    # Broadcasting tensor to all peers 
    def all_reduce(self, tensor: torch.Tensor):
        actions = [{}] 
        if self.rank == 1:
            bt.logging.info('🔔 Action: ALL_REDUCE')
            chunk_shape = (ceil(tensor.size(0) / self.peer_count), ) + tensor.shape[1:]
            action = {
                'type': 'all_reduce', 
                'shape': tuple(tensor.shape),
                'chunk_shape': chunk_shape,
                'dtype': tensor.dtype,
                }
            actions = [action]
            dist.broadcast_object_list(actions, src=1, group = self.validator_rank1)
        dist.broadcast_object_list(actions, src=0)
        action = actions[0]
        tensor_size = tensor.element_size() * tensor.nelement()
        bt.logging.info(f"🔔 Action: {action['type']}  \033[92m{human_readable_size(tensor_size)}\033[0m")
        

        self.reduce_from_peers(tensor)
        del tensor
        gc.collect()
        return self.gather_from_miners(action['shape'], action['dtype'], action['chunk_shape'])

    # Send benchmark request to validator
    def request_benchmark(self):
        
        # Query for Request Benchmark
        query = mapreduce.protocol.RequestBenchmark(
            version=get_my_version(),
            job=mapreduce.protocol.Job(
                rank=self.rank,
                peer_count=self.peer_count,
                bandwidth=self.benchmark_max_size
            ),
        )
        
        set_gloo_socket_ifname(self.dendrite.external_ip)

        # Send query to validator and get job
        response: mapreduce.protocol.RequestBenchmark = self.dendrite.query(
            axons = self.axons[self.validator_uid],
            synapse = query,
            timeout = 30
        )
            
        bt.logging.trace(f"🔵 Response from validator, uid: {self.validator_uid}")
        bt.logging.trace(response)
        
        if response.miner_uid == -1:
            bt.logging.warning(f"No miner to benchmark")
            return False
                
        if response.job.status == "error":
            bt.logging.warning(f"{response.job.reason}")
            return False
        
        if response.job.status is None:
            raise Exception("Unable to connect to validator")
        
        self.miner_uid = response.miner_uid
        self.world_size = response.job.world_size
        self.master_addr = response.job.master_addr
        self.master_port = response.job.master_port
        self.bandwidth = response.job.bandwidth
        
        return True

    # Benchmark miner
    def benchmark(self):
        
        if not self.request_benchmark():
            return False
        
        bt.logging.info(f"⌛ Joining group tcp://{self.master_addr}:{self.master_port} rank: {self.rank}")

        
        bt.logging.info(f'Bandwidth: \033[92m{human_readable_size(self.bandwidth)}\033[0m')
        data_length = int(min(self.bandwidth, self.benchmark_max_size) // 4)
        tensor = self.temp_tensor[:data_length]
        
        dist.init_process_group(
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            backend='gloo',
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=10)
        )
        bt.logging.info('Initialized process group')
        self.init_groups()
        
        actions = [{}] 
        chunk_shape = (ceil(tensor.size(0) / self.peer_count), ) + tensor.shape[1:]
        action = {
            'type': 'all_reduce', 
            'shape': tuple(tensor.shape),
            'chunk_shape': chunk_shape,
            'dtype': tensor.dtype,
            'miner_uid': self.miner_uid
        }
        actions = [action]
        dist.broadcast_object_list(actions, src=1, group = self.validator_rank1)
        dist.broadcast_object_list(actions, src=0)
        action = actions[0]
        tensor_size = tensor.element_size() * tensor.nelement()
        bt.logging.info(f"🔔 Action: {action['type']} \033[92m{human_readable_size(tensor_size)}\033[0m")

        self.reduce_from_peers(tensor)
        del self.temp_tensor
        gc.collect()
        self.gather_from_miners(action['shape'], action['dtype'], action['chunk_shape'])
        return True