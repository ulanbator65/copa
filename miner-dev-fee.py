import json, requests, time, hashlib, string, threading, configparser, os, base64
import re, argparse, configparser
from web3 import Web3
from passlib.hash import argon2
from random import choice, randrange
from json import dumps as json_dumps

import argparse
import configparser

import signal
import json
import sys


def signal_handler(sig, frame):
    global running
    print("Received Ctrl+C. Cleaning up...")
    running = False


signal.signal(signal.SIGINT, signal_handler)

# Set up argument parser
parser = argparse.ArgumentParser(description="Process optional account and worker arguments.")
parser.add_argument('--worker', type=int, help='The worker id to use.')
parser.add_argument('--gpu', type=str, help='Set to true to enable GPU mode, and to false to disable it.')
parser.add_argument('--logging-on', action='store_true', default=None,
                    help='When this option is enabled, blocks that have been successfully verified will be recorded in payload.log')
parser.add_argument('--debug', action='store_true', default=None, help='When this option is enabled, more info output')

# Parse the arguments
args = parser.parse_args()

# Access the arguments via args object
worker_id = args.worker
gpu_mode = args.gpu
logging_on = args.logging_on
debug_output = args.debug

# Load the configuration file
config = configparser.ConfigParser()
config_file_path = 'config.conf'

if os.path.exists(config_file_path):
    config.read(config_file_path)
else:
    raise FileNotFoundError(f"The configuration file {config_file_path} was not found.")

# Ensure that the required settings are present
required_settings = ['difficulty', 'memory_cost', 'cores', 'account', 'last_block_url']
if not all(key in config['Settings'] for key in required_settings):
    missing_keys = [key for key in required_settings if key not in config['Settings']]
    raise KeyError(f"Missing required settings: {', '.join(missing_keys)}")

account = ""
try:
    account = config['Settings']['account_dev_fee']
except:
    sys.exit("Dev fee is disabled")

if args.gpu is not None:
    if args.gpu.lower() == 'true':
        gpu_mode = True
    else:
        gpu_mode = False
else:
    # Ensure that the required settings are present
    if 'gpu_mode' not in config['Settings']:
        missing_keys = [key for key in required_settings if key not in config['Settings']]
        print(f"Missing gpu_mode settings, defaulting to False")
        gpu_mode = False
    else:
        if config['Settings']['gpu_mode'].lower() == 'true':
            gpu_mode = True
        else:
            gpu_mode = False

print(f"\033[93mGPU Mode: {gpu_mode}\033[0m")
if logging_on:
    print("\033[32mLogging verified blocks to payload.log file")
if debug_output:
    print("Output mode: Debug")


def is_valid_ethereum_address(address: str) -> bool:
    # Check if the address matches the basic hexadecimal pattern
    if not re.match("^0x[0-9a-fA-F]{40}$", address):
        return False

    # Check if the address follows EIP-55 checksum encoding
    try:
        # If the checksum is correct, it will return True
        return address == Web3.to_checksum_address(address)
    except ValueError:
        # If a ValueError is raised, the checksum is incorrect
        return False


# Check validity of ethereum address


if is_valid_ethereum_address(account):
    print("The address is valid.  Starting the miner.")
else:
    print("The address is invalid. Correct your account address and try again")
    exit(0)

# Access other settings
difficulty = int(config['Settings']['difficulty'])
memory_cost = int(config['Settings']['memory_cost'])
cores = int(config['Settings']['cores'])
last_block_url = config['Settings']['last_block_url']


def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build_merkle_tree(elements, merkle_tree={}):
    if len(elements) == 1:
        return elements[0], merkle_tree

    new_elements = []
    for i in range(0, len(elements), 2):
        left = elements[i]
        right = elements[i + 1] if i + 1 < len(elements) else left
        combined = left + right
        new_hash = hash_value(combined)
        merkle_tree[new_hash] = {'left': left, 'right': right}
        new_elements.append(new_hash)
    return build_merkle_tree(new_elements, merkle_tree)


from datetime import datetime


def is_within_five_minutes_of_hour():
    timestamp = datetime.now()
    minutes = timestamp.minute
    return 0 <= minutes < 5 or 55 <= minutes < 60


class Block:
    def __init__(self, index, prev_hash, data, valid_hash, random_data, attempts):
        self.index = index
        self.prev_hash = prev_hash
        self.data = data
        self.valid_hash = valid_hash
        self.random_data = random_data
        self.attempts = attempts
        self.timestamp = time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        sha256 = hashlib.sha256()
        sha256.update(f"{self.index}{self.prev_hash}{self.data}{self.valid_hash}{self.timestamp}".encode("utf-8"))
        return sha256.hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "data": self.data,
            "valid_hash": self.valid_hash,
            "random_data": self.random_data,
            "timestamp": self.timestamp,
            "hash": self.hash,
            "attempts": self.attempts
        }


updated_memory_cost = 90000  # just initialize it


def write_difficulty_to_file(difficulty, filename='difficulty.txt'):
    try:
        with open(filename, 'w') as file:
            file.write(difficulty)
    except Exception as e:
        print(f"An error occurred while writing difficulty to file: {e}")


def update_memory_cost_periodically():
    global memory_cost
    global updated_memory_cost
    global gpu_mode
    global running
    time.sleep(2)
    while True:
        if (not running):
            break
        updated_memory_cost = fetch_difficulty_from_server()
        if updated_memory_cost != memory_cost:
            if gpu_mode:
                memory_cost = updated_memory_cost
            print(f"Updating difficulty to {updated_memory_cost}")
        time.sleep(10)


# Function to get difficulty level from the server
def fetch_difficulty_from_server():
    global memory_cost
    try:
        response = requests.get('http://xenblocks.io/difficulty', timeout=10)
        response_data = response.json()
        return str(response_data['difficulty'])
    except Exception as e:
        # if debug_output:
        # print(f"An error occurred while fetching difficulty: {e}")
        return memory_cost  # Return last value if fetching fails


def generate_random_sha256(max_length=128):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(choice(characters) for _ in range(randrange(1, max_length + 1)))

    sha256 = hashlib.sha256()
    sha256.update(random_string.encode('utf-8'))
    return sha256.hexdigest()


from tqdm import tqdm
import time


def submit_pow(account_address, key, hash_to_verify):
    # Download last block record
    url = last_block_url

    try:
        # Attempt to download the last block record
        response = requests.get(url, timeout=10)  # Adding a timeout of 10 seconds
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred: {e}")
        return None  # Optionally return an error value or re-raise the exception

    if response.status_code != 200:
        # Handle unexpected HTTP status codes
        print(f"Unexpected status code {response.status_code}: {response.text}")
        return None  # Optionally return an error value

    if response.status_code == 200:
        records = json.loads(response.text)
        verified_hashes = []

        for record in records:
            block_id = record.get('block_id')
            record_hash_to_verify = record.get('hash_to_verify')
            record_key = record.get('key')
            account = record.get('account')

            # Verify each record using Argon2
            if record_key is None or record_hash_to_verify is None:
                print(
                    f'Skipping record due to None value(s): record_key: {record_key}, record_hash_to_verify: {record_hash_to_verify}')
                continue  # skip to the next record

            if argon2.verify(record_key, record_hash_to_verify):
                verified_hashes.append(hash_value(str(block_id) + record_hash_to_verify + record_key + account))

        # If we have any verified hashes, build the Merkle root
        if verified_hashes:
            merkle_root, _ = build_merkle_tree(verified_hashes)

            # Calculate block ID for output (using the last record for reference)
            output_block_id = int(block_id / 100)

            # Prepare payload for PoW
            payload = {
                'account_address': account_address,
                'block_id': output_block_id,
                'merkle_root': merkle_root,
                'key': key,
                'hash_to_verify': hash_to_verify
            }

            # Send POST request
            try:
                pow_response = requests.post('http://xenblocks.io:4446/send_pow', json=payload)
                if pow_response.status_code == 200:
                    print(f"Proof of Work successful: {pow_response.json()}")
                else:
                    print(f"Proof of Work failed: {pow_response.json()}")

                print(f"Block ID: {output_block_id}, Merkle Root: {merkle_root}")
            except requests.exceptions.RequestException as e:
                # Handle any exceptions that occur during the request
                print(f"An error occurred: {e}")
                return None

    else:
        print("Failed to fetch the last block.")


# ANSI escape codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[36m"
RESET = "\033[0m"

normal_blocks_count = 0
super_blocks_count = 0
xuni_blocks_count = 0


def submit_block(key, account):
    global updated_memory_cost  # Make it global so that we can update it
    found_valid_hash = False

    global normal_blocks_count
    global super_blocks_count
    global xuni_blocks_count

    remove_prefix_address = account[2:]
    salt = bytes.fromhex(remove_prefix_address)

    argon2_hasher = argon2.using(time_cost=difficulty, salt=salt, memory_cost=memory_cost, parallelism=cores,
                                 hash_len=64)

    hashed_data = argon2_hasher.hash(key)
    parts = hashed_data.split("$")
    if len(parts) > 5:
        only_hashed_data = parts[5]
    else:
        print("Invalid hash format")
        return None

    isSuperblock = False

    for target in stored_targets:

        if target in only_hashed_data:
            # Search for the pattern "XUNI" followed by a digit (0-9)
            if re.search("XUNI[0-9]", only_hashed_data) and is_within_five_minutes_of_hour():
                found_valid_hash = True
                break
            elif target == "XEN11":
                found_valid_hash = True
                capital_count = sum(1 for char in re.sub('[0-9]', '', only_hashed_data) if char.isupper())
                if capital_count >= 50:
                    isSuperblock = True
                    print(f"{RED}Superblock found{RESET}")
                break
            else:
                found_valid_hash = False
                break

    if found_valid_hash:
        print(f"\n{RED}Found valid hash for target {target}{RESET}")

        now = datetime.now()  # Get the current time

        # Prepare the payload
        payload = {
            "hash_to_verify": hashed_data,
            "key": key,
            "account": account,
            "attempts": "140000",
            "hashes_per_second": "1000",
            "worker": worker_id  # Adding worker information to the payload
        }

        print(payload)

        if logging_on:
            with open("payload.log", "a") as payload_file:
                payload_file.write(json.dumps(payload) + "\n")

        max_retries = 5
        retries = 0

        while retries <= max_retries:
            try:
                # Make the POST request
                response = requests.post('http://xenblocks.io/verify', json=payload, timeout=10)

                # Print the HTTP status code
                print("HTTP Status Code:", response.status_code)
                # Print the server's response
                print("Server Response:", response.json())
                if found_valid_hash and response.status_code == 200:
                    if "XUNI" in only_hashed_data:
                        xuni_blocks_count += 1
                        break
                    elif "XEN11" in only_hashed_data:
                        capital_count = sum(1 for char in re.sub('[0-9]', '', only_hashed_data) if char.isupper())
                        if capital_count >= 50:
                            super_blocks_count += 1
                        else:
                            normal_blocks_count += 1
                if target == "XEN11" and found_valid_hash and response.status_code == 200:
                    # submit proof of work validation of last sealed block
                    submit_pow(account, key, hashed_data)
                    break
                if response.status_code != 500:  # If status code is not 500, break the loop
                    print("Server Response:", response.json())
                    return None

                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(3)  # You can adjust the sleep time
            except Exception as e:
                print("An error occurred:", e)
        if (retries > max_retries):
            print(f"Failed to submit block after {retries} retries")
            return None
        return key, hashed_data
    return None


gpu_hash_rate_dir = "hash_rates"
EXPIRATION_TIME = 120


def clear_existing_files():
    for filename in os.listdir(gpu_hash_rate_dir):
        filepath = os.path.join(gpu_hash_rate_dir, filename)
        try:
            os.remove(filepath)
        except Exception as e:
            # Print an error message if a file can't be removed.
            print(f"Error removing file {filepath}: {e}")


def get_all_hash_rates():
    total_hash_rate = 0
    active_processes = 0
    current_time = time.time()
    for filename in os.listdir(gpu_hash_rate_dir):
        filepath = os.path.join(gpu_hash_rate_dir, filename)
        try:
            # If a file is older than EXPIRATION_TIME, remove it.
            if current_time - os.path.getmtime(filepath) > EXPIRATION_TIME:
                os.remove(filepath)
                continue

            # Read the hash rate from the file and add it to the total rate.
            with open(filepath, "r") as f:
                hash_rate = float(f.read().strip())
                total_hash_rate += hash_rate

            active_processes += 1
        except (ValueError, IOError) as e:
            # Ignore files with invalid content or that can't be read.
            pass
    return total_hash_rate, active_processes


total_hash_rate = 0
active_processes = 0


def monitor_hash_rate():
    if not os.path.exists(gpu_hash_rate_dir):
        os.makedirs(gpu_hash_rate_dir)
    clear_existing_files()
    global total_hash_rate
    global active_processes
    global running

    while True:
        if (not running):
            break
        total_hash_rate, active_processes = get_all_hash_rates()
        time.sleep(1)


def monitor_blocks_directory(account):
    global normal_blocks_count
    global super_blocks_count
    global xuni_blocks_count
    global memory_cost
    global running
    with tqdm(total=None, dynamic_ncols=True, desc=f"{GREEN}Mining{RESET}", unit=f" {GREEN}Blocks{RESET}") as pbar:
        pbar.update(0)
        while True:
            if (not running):
                break
            try:
                BlockDir = f"gpu_found_blocks_dev_fee_tmp/"
                if not os.path.exists(BlockDir):
                    os.makedirs(BlockDir)
                for filename in os.listdir(BlockDir):
                    filepath = os.path.join(BlockDir, filename)
                    with open(filepath, 'r') as f:
                        data = f.read()
                    if (submit_block(data, account) is not None):
                        pbar.update(1)
                    os.remove(filepath)
                superblock = f"{RED}super:{super_blocks_count}{RESET} "
                block = f"{GREEN}normal:{normal_blocks_count}{RESET} "
                xuni = f"{BLUE}xuni:{xuni_blocks_count}{RESET} "
                if (super_blocks_count == 0):
                    superblock = ""
                if (normal_blocks_count == 0):
                    block = ""
                if (xuni_blocks_count == 0):
                    xuni = ""
                if super_blocks_count == 0 and normal_blocks_count == 0 and xuni_blocks_count == 0:
                    pbar.set_postfix({
                                      "Stat": f"Active:{BLUE}{active_processes}{RESET}, HashRate:{BLUE}{total_hash_rate:.2f}{RESET}h/s",
                                      "Difficulty": f"{YELLOW}{memory_cost}{RESET}"}, refresh=True)
                else:
                    pbar.set_postfix({"Details": f"{superblock}{block}{xuni}",
                                      "Stat": f"Active:{BLUE}{active_processes}{RESET}, HashRate:{BLUE}{total_hash_rate:.2f}{RESET}h/s",
                                      "Difficulty": f"{YELLOW}{memory_cost}{RESET}"}, refresh=True)

                time.sleep(1)
            except Exception as e:
                print(f"An error occurred while monitoring blocks directory: {e}")


if __name__ == "__main__":
    blockchain = []
    stored_targets = ['XEN11', 'XUNI']
    num_blocks_to_mine = 20000000
    global running
    running = True
    updated_memory_cost = fetch_difficulty_from_server()
    if updated_memory_cost != memory_cost:
        if gpu_mode:
            memory_cost = updated_memory_cost
        print(f"Updating difficulty to {updated_memory_cost}")

    # Start difficulty monitoring thread
    difficulty_thread = threading.Thread(target=update_memory_cost_periodically)
    difficulty_thread.daemon = True  # This makes the thread exit when the main program exits
    difficulty_thread.start()

    hashrate_thread = threading.Thread(target=monitor_hash_rate)
    hashrate_thread.daemon = True  # This makes the thread exit when the main program exits
    hashrate_thread.start()

    genesis_block = Block(0, "0", "Genesis Block", "0", "0", "0")
    blockchain.append(genesis_block.to_dict())
    print(f"Mining with: {RED}{account}{RESET}")
    if (gpu_mode):
        print(f"Using GPU mode")
        submit_thread = threading.Thread(target=monitor_blocks_directory, args=(account,))
        submit_thread.daemon = True  # This makes the thread exit when the main program exits
        submit_thread.start()

        try:
            while True:  # Loop forever
                if (not running):
                    break
                time.sleep(2)  # Sleep for 2 seconds
        except KeyboardInterrupt:
            print("Main thread is finished")
