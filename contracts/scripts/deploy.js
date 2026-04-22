const hre = require("hardhat");

async function main() {
  console.log("Deploying VideoForensicLedger to Sepolia...\n");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying from wallet:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Wallet balance:", hre.ethers.formatEther(balance), "ETH\n");

  const Ledger = await hre.ethers.getContractFactory("VideoForensicLedger");
  const ledger = await Ledger.deploy();
  await ledger.waitForDeployment();

  const address = await ledger.getAddress();
  console.log("✅ Contract deployed at:", address);
  console.log("   Explorer: https://sepolia.etherscan.io/address/" + address);
  console.log("\n👉 Copy this into your .env file:");
  console.log("   CONTRACT_ADDRESS=" + address);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
