// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title VideoForensicLedger
 * @author Your Team
 * @notice Stores SHA-256 fingerprint hashes of registered videos on-chain.
 *         Once registered, a hash cannot be modified — providing a tamper-proof
 *         record for forensic verification and insurance claims.
 *
 * Deployment:
 *   - Network  : Sepolia testnet (chain ID 11155111)
 *   - Compiler : solc ^0.8.20
 *   - Tool     : Hardhat  (`npx hardhat run scripts/deploy.js --network sepolia`)
 *
 * How it works:
 *   1. Backend calls storeHash(videoId, hash) after processing a video.
 *   2. The hash is stored permanently in the mapping.
 *   3. Backend calls getHash(videoId) during verification.
 *   4. If the hash is found and matches, the video is authentic.
 */
contract VideoForensicLedger {

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    address public owner;

    struct VideoRecord {
        string  hash;          // SHA-256 hex digest of the AI fingerprint
        uint256 timestamp;     // Block timestamp at registration
        address registeredBy;  // Wallet that submitted the registration
        bool    exists;        // Guard flag — false means "not registered"
    }

    // videoId (off-chain MD5 of filename) => VideoRecord
    mapping(string => VideoRecord) private records;

    // -----------------------------------------------------------------------
    // Events  (indexed fields are filterable in web3 event queries)
    // -----------------------------------------------------------------------

    /// Emitted when a new video hash is registered.
    event HashRegistered(
        string  indexed videoId,
        string  hash,
        address indexed registeredBy,
        uint256 timestamp
    );

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorised: caller is not owner");
        _;
    }

    modifier notAlreadyRegistered(string memory videoId) {
        require(!records[videoId].exists, "Video already registered");
        _;
    }

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    constructor() {
        owner = msg.sender;
    }

    // -----------------------------------------------------------------------
    // Write functions
    // -----------------------------------------------------------------------

    /**
     * @notice Register a video's SHA-256 fingerprint hash on-chain.
     * @dev    Can only be called once per videoId — hashes are immutable.
     * @param  videoId  Unique identifier for the video (MD5 of filename, 12 chars)
     * @param  hash     SHA-256 hex digest of the AI-generated fingerprint vector
     */
    function storeHash(string memory videoId, string memory hash)
        external
        notAlreadyRegistered(videoId)
    {
        require(bytes(videoId).length > 0, "videoId cannot be empty");
        require(bytes(hash).length == 64,  "hash must be 64-char SHA-256 hex");

        records[videoId] = VideoRecord({
            hash:         hash,
            timestamp:    block.timestamp,
            registeredBy: msg.sender,
            exists:       true
        });

        emit HashRegistered(videoId, hash, msg.sender, block.timestamp);
    }

    /**
     * @notice Retrieve the stored hash for a video.
     * @param  videoId  The video's unique identifier.
     * @return hash     SHA-256 hex digest, or empty string if not found.
     * @return timestamp  Unix timestamp of registration.
     * @return registeredBy  Address that registered the video.
     * @return exists    True if the video has been registered.
     */
    function getRecord(string memory videoId)
        external
        view
        returns (
            string  memory hash,
            uint256        timestamp,
            address        registeredBy,
            bool           exists
        )
    {
        VideoRecord memory r = records[videoId];
        return (r.hash, r.timestamp, r.registeredBy, r.exists);
    }

    /**
     * @notice Quick hash-only lookup (cheaper gas for verification calls).
     * @param  videoId  The video's unique identifier.
     * @return          SHA-256 hex digest, or empty string if not found.
     */
    function getHash(string memory videoId)
        external
        view
        returns (string memory)
    {
        return records[videoId].hash;
    }

    /**
     * @notice Check whether a video has been registered.
     * @param  videoId  The video's unique identifier.
     */
    function isRegistered(string memory videoId) external view returns (bool) {
        return records[videoId].exists;
    }
}
