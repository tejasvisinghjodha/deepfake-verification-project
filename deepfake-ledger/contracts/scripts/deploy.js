// contracts/scripts/deploy.js
// Run with: npx hardhat run contracts/scripts/deploy.js --network sepolia

const hre = require("hardhat");

async function main() {
  console.log("Deploying VideoForensicLedger to Sepolia testnet...\n");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying from wallet:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Wallet balance:", hre.ethers.utils.formatEther(balance), "ETH\n");

  // Deploy the contract
  const Ledger = await hre.ethers.getContractFactory("VideoForensicLedger");
  const ledger = await Ledger.deploy();
  await ledger.deployed(); // ethers v5 fix

  console.log("✅ Contract deployed at:", ledger.address);
  console.log("   Explorer: https://sepolia.etherscan.io/address/" + ledger.address);
  console.log("\n👉 Add this to your .env file:");
  console.log("   CONTRACT_ADDRESS=" + ledger.address);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});