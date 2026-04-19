from web3 import Web3
from eth_account import Account
import os

w3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER_URL")))
acct = Account.from_key(os.getenv("PRIVATE_KEY"))

print("=== DEBUG WALLET ===")
print("Address:", acct.address)
print("Balance:", w3.from_wei(w3.eth.get_balance(acct.address), "ether"))
print("====================")