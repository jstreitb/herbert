package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * The mandatory first line of every session JSONL file. Identifies the schema version, mod
 * build, and session, and carries a privacy-preserving hash of the player's identity instead of
 * their raw username.
 */
public final class SessionHeader {

    /** Semantic version of the per-tick JSONL schema (see {@link dev.herbert.bridgelogger.util.HerbertConstants#SCHEMA_VERSION}). */
    @SerializedName("schema_version")
    public final String schemaVersion;

    /** Version of the BridgeLogger mod that produced this file. */
    @SerializedName("herbert_mod_version")
    public final String herbertModVersion;

    /** Random UUID (v4) uniquely identifying this recording session. */
    @SerializedName("session_id")
    public final String sessionId;

    /** ISO-8601 UTC timestamp marking when recording started. */
    @SerializedName("recording_start_timestamp")
    public final String recordingStartTimestamp;

    /**
     * SHA-256 hex digest of the local player's username. The raw username is never written
     * anywhere in this file or any log produced by BridgeLogger; see
     * {@link dev.herbert.bridgelogger.util.HashUtil}.
     */
    @SerializedName("player_username_hash")
    public final String playerUsernameHash;

    /**
     * Creates an immutable session header.
     *
     * @param schemaVersion semver string of the JSONL schema
     * @param herbertModVersion version of the mod that produced this file
     * @param sessionId UUID string identifying this session
     * @param recordingStartTimestamp ISO-8601 UTC start timestamp
     * @param playerUsernameHash SHA-256 hex digest of the player's username; never the raw username
     */
    public SessionHeader(String schemaVersion, String herbertModVersion, String sessionId,
            String recordingStartTimestamp, String playerUsernameHash) {
        this.schemaVersion = schemaVersion;
        this.herbertModVersion = herbertModVersion;
        this.sessionId = sessionId;
        this.recordingStartTimestamp = recordingStartTimestamp;
        this.playerUsernameHash = playerUsernameHash;
    }
}
