# utils.py
# Utility Functions and Classes for MapReduce Operations
# The `utils.py` module in this MapReduce application contains a series of helper functions and classes that provide common, reusable functionalities required by different components of the application. These utilities streamline tasks such as data formatting, network communication, version checking, and memory management. 
# The use of a dedicated utility 
# module promotes code reusability and modularity, enabling a cleaner, more maintainable, and efficient codebase. Functions in this module are designed to be generic and abstract enough to be used in various parts of the application without modification, illustrating a key principle of DRY (Don't Repeat Yourself).

import bittensor as bt
import os
import time
import mapreduce
import netifaces
import torch
import socket
import psutil
import random

"""
Converts a size in bytes to a more human-readable format (e.g., KB, MB).

Args:
    size (int): The size in bytes.
    decimal_places (int, optional): Number of decimal places for the size. Defaults to 2.

Returns:
    str: Human-readable size string.
"""
def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

"""
Calculates the size of a specific chunk given the total file size and number of chunks.

Args:
    file_size (int): Total size of the file.
    chunk_count (int): Total number of chunks to divide the file into.
    chunk_id (int): The specific chunk ID for which size is calculated.

Returns:
    int: Size of the specified chunk.
"""
def get_chunk_size(file_size, chunk_count, chunk_id):
    chunk_size = file_size // chunk_count
    if chunk_id == chunk_count - 1:
        chunk_size += file_size % chunk_count
    return chunk_size
    
"""
Splits a tensor into chunks with padding added to the last chunk if needed.

Args:
    tensor (Tensor): The tensor to be chunked.
    num_chunks (int): Number of chunks to split the tensor into.
    dim (int, optional): The dimension along which to split the tensor. Defaults to 0.

Returns:
    List[Tensor]: List of tensor chunks.
"""
def chunk_with_padding(tensor, num_chunks, dim=0):
    # Determine the size of each chunk
    chunk_size = (tensor.size(dim) + num_chunks - 1) // num_chunks  # Ceiling division

    # Split the tensor into chunks
    chunks = list(tensor.chunk(num_chunks, dim=dim))  # Convert to list for mutability
    
    # Pad the last chunk if necessary
    last_chunk = chunks[-1]
    if last_chunk.size(dim) < chunk_size:
        # Calculate the padding size
        pad_size = chunk_size - last_chunk.size(dim)
        
        # Create a padding tensor of zeros
        pad_tensor = torch.zeros(*last_chunk.shape[:dim], pad_size, *last_chunk.shape[dim+1:], device=tensor.device, dtype=tensor.dtype)
        
        # Concatenate the padding tensor to the last chunk
        chunks[-1] = torch.cat([last_chunk, pad_tensor], dim=dim)  # Now this line will work
    
    return chunks  # You can convert it back to a tuple if necessary by `return tuple(chunks)`

"""
Merges a list of chunks back into a single tensor.

Args:
    chunks (List[Tensor]): List of tensor chunks to merge.
    original_size (int): The size of the original tensor before chunking.
    dim (int, optional): The dimension along which to merge the chunks. Defaults to 0.

Returns:
    Tensor: The merged tensor.
"""
def merge_chunks(chunks, original_size, dim=0):
    # Concatenate the chunks along the given dimension
    merged_tensor = torch.cat(chunks, dim=dim)
    # Trim the tensor to the original size
    merged_tensor = merged_tensor.narrow(dim, 0, original_size)
    return merged_tensor

"""
Retrieves the current version of the MapReduce protocol being used.

Returns:
    mapreduce.protocol.Version: The version object with major, minor, and patch components.
"""
def get_my_version() -> mapreduce.protocol.Version:
    version_str = mapreduce.__version__
    major, minor, patch = version_str.split('.')
    return mapreduce.protocol.Version(
        major_version = int(major),
        minor_version = int(minor),
        patch_version = int(patch)
    )

"""
Checks if the provided version matches the current MapReduce protocol version.

Args:
    version (mapreduce.protocol.Version): The version to check.

Returns:
    bool: True if the versions match, False otherwise.
"""
def check_version( version: mapreduce.protocol.Version ) -> bool:
        version_str = mapreduce.__version__
        major, minor, patch = version_str.split('.')
        validator_version_str = f"{version.major_version}.{version.minor_version}.{version.patch_version}"
        if version.major_version != int(major):
            bt.logging.error("🔴 Major version mismatch", f"miner: {version_str}, validator: {validator_version_str}")
            return False
        elif version.minor_version != int(minor):
            bt.logging.warning("🟡 Minor version mismatch", f"miner: {version_str}, validator: {validator_version_str}")
        elif version.patch_version != int(patch):
            bt.logging.warning("🔵 Patch version mismatch", f"miner: {version_str}, validator: {validator_version_str}")
        return True

"""
Checks the status of running processes and updates their status accordingly.

Args:
    processes (dict): Dictionary containing process information.
    miner_status (dict, optional): Dictionary containing the status of miners.
"""
def check_processes(processes, miner_status = None):
    while True:
        keys_to_delete = []
        for key in processes:
            process = processes[key]['process']
            if not process.is_alive():
                keys_to_delete.append(key)
        for key in keys_to_delete:
            bt.logging.info(f"🚩 Delete Process with key: {key}")
            if miner_status and processes[key]['benchmarking']:
                miner_uid = int(processes[key]['miners'][0][0])
                if miner_status[miner_uid]['status'] == 'benchmarking':
                    miner_status[miner_uid]['status'] = 'unavailable'
                    miner_status[miner_uid]['retry'] = miner_status[miner_uid].get('retry', 0) + 1
                    bt.logging.warning(f"Benchmark for Miner {miner_uid} Retry: {miner_status[miner_uid]['retry']}")
                    if miner_status[miner_uid]['retry'] > 3:
                        miner_status[miner_uid] = 'failed'
            if 'miners' in processes[key]:
                for (uid, _) in processes[key]['miners']:
                    if miner_status[int(uid)]['status'] == 'working':
                        miner_status[int(uid)]['status'] = 'available'
            del processes[key]
        time.sleep(1)
        
"""
Checks if any process in the given dictionary is still alive.

Args:
    processes (dict): Dictionary containing process information.

Returns:
    bool: True if any process is alive, False otherwise.
"""
def is_process_running(processes):
    for key in processes:
        process = processes[key]['process']
        if process.is_alive():
            return True
    return False

"""
Finds the network interface associated with a given IP address.

Args:
    target_ip (str): The target IP address.

Returns:
    str: The name of the network interface.

Raises:
    ValueError: If no interface is found for the given IP.
"""
def find_network_interface(target_ip):
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            for addr_info in addrs[netifaces.AF_INET]:
                if addr_info['addr'] == target_ip:
                    return interface
    raise ValueError(f"No interface found for IP {target_ip}")

"""
Sets the network interface for Gloo (used in distributed operations) based on the external IP.

Args:
    external_ip (str): The external IP address.
"""
def set_gloo_socket_ifname(external_ip):
    ifname = find_network_interface(external_ip)
    bt.logging.info(f"IP: {external_ip} IFNAME: {ifname}")
    os.environ['GLOO_SOCKET_IFNAME'] = ifname

"""
Finds an unused port within a specified range.

Args:
    start_port (int): The starting port number.
    end_port (int): The ending port number.

Returns:
    int: An unused port number within the specified range.

Raises:
    Exception: If no unused port is found.
"""
def get_unused_port(start_port, end_port):
    """
    This function finds unused ports in the given range.

    Args:
    start_port (int): The starting port number.
    end_port (int): The ending port number.

    Returns:
    List[int]: A list of unused ports within the given range.
    """
    unused_ports = []
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            res = s.connect_ex(('localhost', port))
            if res != 0:
                unused_ports.append(port)
    if len(unused_ports) == 0:
        raise Exception("No unused ports found")
    #  select random port in unused ports
    return unused_ports[random.randint(0, len(unused_ports) - 1)]


"""
Retrieves the amount of available bandwidth from the free memory.
"""
def calc_bandwidth_from_memory(free_memory: int):
    return max(int((free_memory - 500 * 1024 * 1024) / 2), 0)

"""
Retrieves the amount of free memory in the system.

Returns:
    int: Free memory in bytes.
"""
def get_available_memory():
    # Get the memory details
    memory = psutil.virtual_memory()
    # Free memory in bytes
    return memory.free
