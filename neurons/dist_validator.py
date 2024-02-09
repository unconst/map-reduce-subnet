import bittensor as bt
import time
from threading import Lock, Thread
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from datetime import timedelta
import traceback
from mapreduce.utils import get_my_version, set_gloo_socket_ifname, human_readable_size
import mapreduce
import signal

# Validator class is responsible for validating the miners and managing the mining process.
class Validator:
    # Initialize the validator with the given parameters.
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, wallet: bt.wallet, miners, job: mapreduce.protocol.Job, is_master = False, is_benchmark = False):
        log =  (f'Master Hotkey: {job.master_hotkey} | '\
                f'Client Hotkey: {job.client_hotkey} | '\
                f'Master Addr: {job.master_addr} | '\
                f'Master Port: {job.master_port} | '\
                f'World Size: {job.world_size} | '\
                f'Peer Count: {job.peer_count } | '\
                f'Bandwidth: {job.bandwidth} | '\
                f'Session Time: {job.session_time}')
        bt.logging.info(log)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.miners = miners
        self.job = job
        self.master_addr = job.master_addr
        self.master_port = job.master_port
        self.monitor_interval = 1 # Interval in seconds to monitor peers
        self.rank = 0
        self.is_master = is_master
        self.peer_count = job.peer_count
        self.miner_count = job.peer_count
        
        self.is_benchmark = is_benchmark

        self.lock = Lock()
        self.validators =  {
        }
        self.verifiers = []
        self.master_status = 'not initialized' # 'not initialized', 'asking verifiers', 'waiting for join', 'running', 'stopped', 'error'
        self.error_code = 0
        self.error_message = ''
        self.hotkey = job.master_hotkey
        self.client_hotkey = job.client_hotkey
        
    # Start the master process for the validator.
    def start_master(self): 
        job_sent = False
        try: 
            self.world_size = len(self.verifiers) + self.miner_count + self.peer_count + 1  # Total size of the process group
            # This validator is the moderator
            self.start_miners()
            self.job.world_size = self.world_size
            self.output_queue.put(self.job)
            job_sent = True
            # Start the init task as a background thread
            self.init_master_process_group()
            # Start monitoring peers
            self.monitor_actions()
        except Exception as e:
            bt.logging.error(f"❌ {e}")
            traceback.print_exc()
            if not job_sent:
                self.job.status = 'error'
                self.job.reason = str(e)
                self.output_queue.put(self.job)

    # Initialize the master process group.
    def init_master_process_group(self):
        # lock validator when intializing process group
        # Initialized only it's not initialized before
        if not dist.is_initialized():
            if self.lock.locked(): return
            with self.lock:
                set_gloo_socket_ifname(self.master_addr)
                # Start the init task as a background thread
                bt.logging.info(f"⌛ Waiting for joining  tcp://{self.master_addr}:{self.master_port} rank: {self.rank}, worldsize: {self.world_size}")

                dist.init_process_group(
                    init_method=f"tcp://{self.master_addr}:{self.master_port}",
                    backend='gloo',
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=timedelta(seconds=10)
                )

                bt.logging.info(f"Process group initialized.")
                # Init groups
                self.init_groups()

    # Initialize the groups for the validator.
    def init_groups(self):
        bt.logging.info("Creating groups")
        self.peers = dist.new_group(ranks=list(range(1, self.peer_count + 1)), timeout=timedelta(seconds=10))
        self.miners_group = dist.new_group(ranks=list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1)), timeout=timedelta(seconds=10))
        self.peer_miners = [dist.new_group(ranks=[peer_rank] + list(range(self.peer_count + 1, self.peer_count + self.peer_count + 1)), timeout=timedelta(seconds=10)) for peer_rank in range(1, self.peer_count + 1)]
        self.miner_peers = [dist.new_group(ranks=[miner_rank] + list(range(1, self.peer_count + 1)), timeout=timedelta(seconds=10)) for miner_rank in range(self.peer_count + 1, self.peer_count + self.peer_count + 1)]
        self.validator_rank1 = dist.new_group(ranks=(0, 1), timeout=timedelta(seconds=15))
        bt.logging.info('Groups created')

    # Start the miners for the validator.
    def start_miners(self):
        self.available_miners = []
        
        axons = [ axon for (_uid, axon) in self.miners]
        
        bt.logging.info("⌛ Asking miners for status")
        bt.logging.trace(self.miners)
        responses = self.dendrite.query(axons, mapreduce.protocol.MinerStatus(
            version=get_my_version(),
        ))
        bt.logging.info("response from miners")
        bt.logging.trace(responses)
        for miner, response in zip(self.miners, responses):
            if response.available:
                self.available_miners.append(miner)
        if len(self.available_miners) < self.peer_count:
            # bt.logging.warning(f"Not enough miners available, available miners: {len(self.available_miners)}")
            raise Exception(f"Not enough miners available, available miners: {len(self.available_miners)}")
        bt.logging.info(f"Available Miners: {len(self.available_miners)}")
        bt.logging.trace(self.available_miners)
        
        miner_job = mapreduce.protocol.Job(
            rank=None,
            peer_count=self.peer_count,
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            client_hotkey=self.client_hotkey,
            miner_hotkey=self.hotkey
        )
        ranks = {}
        
        self.available_miners = self.available_miners[:self.peer_count]
        for rank, (uid, _axon) in enumerate(self.available_miners, start = self.peer_count + 1):
            ranks[uid] = rank
            
        join = mapreduce.protocol.Join(
            version=get_my_version(),
            job=miner_job,
            ranks=ranks
        )
        self.job.miners = [ (int(uid), _axon) for (uid, _axon) in self.available_miners]
        
        axons = [ axon for (uid, axon) in self.available_miners]
        responses = self.dendrite.query(axons, join)
        bt.logging.info("Responses from miners")
        bt.logging.trace(responses)
        
    # Monitor the actions of the validator.
    def monitor_actions(self):
        bt.logging.info('⌛ Monitor actions')
        self.job.started_at = time.time()
        while True:
            actions = [{}]
            dist.broadcast_object_list(actions, src=1, group = self.validator_rank1)
            action = actions[0]
            
            bt.logging.info(f"🔔 ACTION {action['type']}")
            bt.logging.trace(action)                
            
            
            if self.is_benchmark:
                if action['type'] == 'all_reduce':
                    # calculate size of job in bytes - action['shape'] - torch shape, action['dtype'] - torch dtype
                    self.job_size = int(action['dtype'].itemsize * torch.prod(torch.tensor(action['shape'])))
                    self.benchmark_start = time.time()
                    self.miner_uid = action['miner_uid']
                    del actions[0]['miner_uid']
            if action['type'] == 'all_reduce' or action['type'] == 'broadcast':
                self.action_size = int(action['dtype'].itemsize * torch.prod(torch.tensor(action['shape'])))
                self.action_start = time.time()
                bt.logging.info(f"Data Transfer: {human_readable_size(self.action_size)}")
            
            if time.time() - self.job.started_at > self.job.session_time:
                action = {'type': 'exit', 'success': False, 'reason': 'session timeout'}
                actions = [action]
            dist.broadcast_object_list(actions, src=0)
            
            if action['type'] == 'exit':
                if self.is_benchmark:
                    self.benchmark_end = time.time()
                    self.benchmark_time = self.benchmark_end - self.benchmark_start
                    log =  (f'🟢 Benchmark Success | UID: {self.miner_uid} | '\
                            f'Duration: {self.benchmark_time:.2f}s | '\
                            f'Data: {human_readable_size(2 * self.job_size)} | '\
                            f'Speed: {human_readable_size(2 * self.job_size // (self.benchmark_time))}/s | '\
                    )
                    # bt.logging.success(log)
                    benchmark_result = mapreduce.protocol.BenchmarkResult(
                        bandwidth = self.job_size,
                        speed = 2 * self.job_size // (self.benchmark_time),
                        duration = self.benchmark_time,
                        data_length = 2 * self.job_size,
                    )
                    self.output_queue.put(benchmark_result)
                    self.input_queue.get()
                bt.logging.success('🟢 Received exit signal')
                dist.destroy_process_group()
                break

def timeout_handler():
    """
    Handler for the alarm signal.
    """
    bt.logging.error("Session Timeout")
    raise Exception("Timeout")

# Start the master process for the Validator.
def start_master_process(input_queue: mp.Queue, output_queue: mp.Queue, wallet: bt.wallet, miners, job: mapreduce.protocol.Job, is_benchmark = False):
    """
    Starts the master process for the Validator.

    Args:
        input_queue (mp.Queue): Queue for input messages.
        output_queue (mp.Queue): Queue for output messages.
        wallet (bt.wallet): Wallet instance for transactions.
        miners (list): List of miners participating in the process.
        job (mapreduce.protocol.Job): Job information.
        is_benchmark (bool, optional): Flag to indicate if this process is for benchmarking. Defaults to False.
    """
    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule the alarm to go off after session time
    signal.alarm(job.session_time)

    validator = Validator(input_queue, output_queue, wallet, miners, job, is_master = True, is_benchmark = is_benchmark)
    try:
        validator.start_master()
    except Exception as e:
        bt.logging.info(f"❌ {e}")
        traceback.print_exc()
