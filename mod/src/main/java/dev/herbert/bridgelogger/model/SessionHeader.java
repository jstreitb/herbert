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
     * SHA-256 hex digest of the local player's username. Always present, regardless of the
     * player's answer to the session-end display-name prompt (see {@link #playerUsernameDisplay}),
     * so every session file can be deduplicated/grouped per contributor even when the raw
     * username was not shared; see {@link dev.herbert.bridgelogger.util.HashUtil}.
     */
    @SerializedName("player_username_hash")
    public final String playerUsernameHash;

    /**
     * The player's raw (plaintext) username, present <b>only</b> if the player explicitly opted
     * in via the session-end "Display your username publicly? [y/n]" chat prompt (see
     * {@link dev.herbert.bridgelogger.session.SessionManager}). {@code null} whenever the player
     * answered "n", let the prompt time out, or entered invalid input twice -- in every one of
     * those cases this field is omitted from the serialized JSON entirely (never written as
     * {@code null} or an empty string), since {@link dev.herbert.bridgelogger.serialize.SessionSerializer}
     * uses a Gson instance that drops {@code null} fields rather than emitting them. Added in
     * schema {@code 1.1.0}; absent in every {@code 1.0.0} session file.
     */
    @SerializedName("player_username_display")
    public final String playerUsernameDisplay;

    /**
     * Creates an immutable session header.
     *
     * @param schemaVersion semver string of the JSONL schema
     * @param herbertModVersion version of the mod that produced this file
     * @param sessionId UUID string identifying this session
     * @param recordingStartTimestamp ISO-8601 UTC start timestamp
     * @param playerUsernameHash SHA-256 hex digest of the player's username; never the raw username
     * @param playerUsernameDisplay the player's raw username, or {@code null} if they did not opt
     *        in to displaying it publicly (see {@link #playerUsernameDisplay})
     */
    public SessionHeader(String schemaVersion, String herbertModVersion, String sessionId,
            String recordingStartTimestamp, String playerUsernameHash, String playerUsernameDisplay) {
        this.schemaVersion = schemaVersion;
        this.herbertModVersion = herbertModVersion;
        this.sessionId = sessionId;
        this.recordingStartTimestamp = recordingStartTimestamp;
        this.playerUsernameHash = playerUsernameHash;
        this.playerUsernameDisplay = playerUsernameDisplay;
    }
}
