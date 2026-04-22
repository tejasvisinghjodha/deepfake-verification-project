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
├── app.py                         # Flask API 
├── model/
│   ├── video_processor.py         # OpenCV frame extraction 
│   └── ai_fingerprinter.py        # ResNet-50 fingerprinting 
├── contracts/
│   ├── VideoForensicLedger.sol    # Solidity contract 
│   ├── blockchain_client.py       # web3.py client 
│   └── scripts/deploy.js
├── templates/index.html           # Frontend UI 
├── requirements.txt
├── hardhat.config.js
└── .env.example
```

## API
- `POST /api/register` — upload video, fingerprint, store hash on blockchain
- `POST /api/verify`   — re-fingerprint video and compare against chain record
- `GET  /api/lookup/<video_id>` — fetch stored hash


## Pipeline
Video → OpenCV frames → ResNet-50 features → mean pool → SHA-256 → Ethereum
