"""
Member 3 — Blockchain Client
==============================
Python interface to the VideoForensicLedger smart contract.
Uses web3.py to interact with the Sepolia testnet.

Setup:
  1. Copy .env.example to .env and fill in your values:
       WEB3_PROVIDER_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
       PRIVATE_KEY=0xYOUR_WALLET_PRIVATE_KEY
       CONTRACT_ADDRESS=0xDEPLOYED_CONTRACT_ADDRESS

  2. Deploy the contract first:
       cd contracts && npx hardhat run scripts/deploy.js --network sepolia

  3. Paste the deployed address into .env as CONTRACT_ADDRESS.

Usage:
    from contracts.blockchain_client import BlockchainClient
    client = BlockchainClient()
    receipt = client.store_hash("abc123def456", "sha256hexdigest...")
    stored  = client.get_hash("abc123def456")
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ABI — only the functions we call (reduces import size vs full ABI)
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "videoId", "type": "string"},
            {"internalType": "string", "name": "hash",    "type": "string"},
        ],
        "name": "storeHash",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "videoId", "type": "string"}],
        "name": "getHash",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "videoId", "type": "string"}],
        "name": "getRecord",
        "outputs": [
            {"internalType": "string",  "name": "hash",         "type": "string"},
            {"internalType": "uint256", "name": "timestamp",    "type": "uint256"},
            {"internalType": "address", "name": "registeredBy", "type": "address"},
            {"internalType": "bool",    "name": "exists",       "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "videoId", "type": "string"}],
        "name": "isRegistered",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "name": "videoId",      "type": "string"},
            {"indexed": False, "name": "hash",          "type": "string"},
            {"indexed": True,  "name": "registeredBy",  "type": "address"},
            {"indexed": False, "name": "timestamp",     "type": "uint256"},
        ],
        "name": "HashRegistered",
        "type": "event",
    },
]

SEPOLIA_EXPLORER = "https://sepolia.etherscan.io/tx/"


class BlockchainClient:
    """
    Thin wrapper around web3.py for interacting with the deployed
    VideoForensicLedger contract on Sepolia testnet.
    """

    def __init__(self):
        from web3 import Web3

        provider_url = os.getenv("WEB3_PROVIDER_URL")
        private_key  = os.getenv("PRIVATE_KEY")
        contract_addr = os.getenv("CONTRACT_ADDRESS")

        if not all([provider_url, private_key, contract_addr]):
            raise EnvironmentError(
                "Missing environment variables. "
                "Check WEB3_PROVIDER_URL, PRIVATE_KEY, CONTRACT_ADDRESS in .env"
            )

        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to Ethereum node at {provider_url}")

        self.account = self.w3.eth.account.from_key(private_key)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_addr),
            abi=CONTRACT_ABI,
        )

        print(f"[BlockchainClient] Connected to Sepolia via {provider_url[:40]}...")
        print(f"[BlockchainClient] Wallet: {self.account.address}")
        print(f"[BlockchainClient] Contract: {contract_addr}")

    # ------------------------------------------------------------------
    # Write operations (cost gas — require funded wallet)
    # ------------------------------------------------------------------

    def store_hash(self, video_id: str, video_hash: str) -> dict:
        """
        Call storeHash() on the smart contract and wait for confirmation.

        Parameters
        ----------
        video_id   : str  12-char unique ID for the video
        video_hash : str  64-char SHA-256 hex digest

        Returns
        -------
        dict with tx_hash, block_number, explorer_url
        """
        nonce = self.w3.eth.get_transaction_count(self.account.address)

        # Build the transaction
        txn = self.contract.functions.storeHash(video_id, video_hash).build_transaction({
            "from":     self.account.address,
            "nonce":    nonce,
            "gas":      200_000,
            "gasPrice": self.w3.eth.gas_price,
        })

        # Sign with private key
        signed_txn = self.w3.eth.account.sign_transaction(txn, private_key=os.getenv("PRIVATE_KEY"))

        # Broadcast to network
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)

        # Wait for the transaction to be mined (up to 120s)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] != 1:
            raise RuntimeError(f"Transaction failed: {tx_hash.hex()}")

        return {
            "tx_hash":      tx_hash.hex(),
            "block_number": receipt["blockNumber"],
            "explorer_url": SEPOLIA_EXPLORER + tx_hash.hex(),
        }

    # ------------------------------------------------------------------
    # Read operations (free — no gas)
    # ------------------------------------------------------------------

    def get_hash(self, video_id: str) -> str | None:
        """
        Fetch the stored SHA-256 hash for a video from the blockchain.

        Returns the 64-char hex string, or None if not registered.
        """
        stored = self.contract.functions.getHash(video_id).call()
        return stored if stored else None

    def get_record(self, video_id: str) -> dict | None:
        """
        Fetch the full VideoRecord struct for a given video_id.
        Returns None if not registered.
        """
        hash_, timestamp, registered_by, exists = (
            self.contract.functions.getRecord(video_id).call()
        )
        if not exists:
            return None
        return {
            "hash":          hash_,
            "timestamp":     timestamp,
            "registered_by": registered_by,
            "exists":        exists,
        }

    def is_registered(self, video_id: str) -> bool:
        return self.contract.functions.isRegistered(video_id).call()
