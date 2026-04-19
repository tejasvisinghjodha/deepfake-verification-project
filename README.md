# AI-Driven Deepfake Insurance & Forensic Ledger

A full-stack prototype: AI fingerprinting + Ethereum blockchain for video authenticity verification.

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in your keys
npm install                   # for Hardhat
npx hardhat run contracts/scripts/deploy.js --network sepolia
python app.py                 # open http://localhost:5000
```

## Structure
```
deepfake-ledger/
├── app.py                         # Flask API (Member 4)
├── model/
│   ├── video_processor.py         # OpenCV frame extraction (Member 1)
│   └── ai_fingerprinter.py        # ResNet-50 fingerprinting (Member 2)
├── contracts/
│   ├── VideoForensicLedger.sol    # Solidity contract (Member 3)
│   ├── blockchain_client.py       # web3.py client (Member 3)
│   └── scripts/deploy.js
├── templates/index.html           # Frontend UI (Member 4)
├── requirements.txt
├── hardhat.config.js
└── .env.example
```

## API
- `POST /api/register` — upload video, fingerprint, store hash on blockchain
- `POST /api/verify`   — re-fingerprint video and compare against chain record
- `GET  /api/lookup/<video_id>` — fetch stored hash

## Blockchain Setup (Member 3)
1. MetaMask → switch to Sepolia testnet
2. Get test ETH: sepoliafaucet.com
3. Infura.io → new project → copy Sepolia RPC URL
4. Fill `.env` with `WEB3_PROVIDER_URL`, `PRIVATE_KEY`, then deploy
5. Copy printed `CONTRACT_ADDRESS` into `.env`

## Pipeline
Video → OpenCV frames → ResNet-50 features → mean pool → SHA-256 → Ethereum
