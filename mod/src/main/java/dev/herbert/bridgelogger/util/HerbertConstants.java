package dev.herbert.bridgelogger.util;

/**
 * Central location for every named constant used across the BridgeLogger mod.
 *
 * <p>Per the project quality bar, no class in this mod should contain "magic numbers" or
 * inline literal defaults scattered through the codebase. Every tunable or fixed value that
 * is not exposed as a user-facing {@link dev.herbert.bridgelogger.config.HerbertConfig} entry
 * lives here instead, fully documented.</p>
 */
public final class HerbertConstants {

    /** Forge mod id. Must match {@code gradle.properties} and {@code mcmod.info}. */
    public static final String MOD_ID = "bridgelogger";

    /** Human readable mod name. */
    public static final String MOD_NAME = "BridgeLogger";

    /** Current mod build version. Also reported in the session header as {@code herbert_mod_version}. */
    public static final String MOD_VERSION = "1.0.0";

    /**
     * Semantic version of the JSONL schema written by {@link dev.herbert.bridgelogger.serialize.SessionSerializer}.
     *
     * <p>Bumped to {@code 1.2.0} (from {@code 1.1.0}) for the addition of the optional
     * {@code chunk_index}/{@code chunk_total} session-header fields (see
     * {@link dev.herbert.bridgelogger.serialize.SessionChunker}) -- another backward-compatible,
     * additive change: both fields are only ever present on the header of a chunked upload's
     * split files, never on a single-file session, and are added by copying the original header
     * line byte-for-byte and appending the two fields to that copy (see
     * {@link dev.herbert.bridgelogger.serialize.SessionChunker#withChunkMetadata(String, int, int)}),
     * never by re-serializing the header object. (Previously bumped to {@code 1.1.0} for the
     * optional {@code player_username_display} field -- see
     * {@link dev.herbert.bridgelogger.model.SessionHeader}.) See {@code mod/README.md}'s JSONL
     * schema section.</p>
     */
    public static final String SCHEMA_VERSION = "1.2.0";

    /** Vanilla Minecraft tick rate, used only for documentation/derivations (ticks per second). */
    public static final int VANILLA_TICKS_PER_SECOND = 20;

    /** Default width (X axis, blocks) of the local block grid sampled around the player. */
    public static final int DEFAULT_GRID_WIDTH = 7;

    /** Default height (Y axis, blocks) of the local block grid sampled around the player. */
    public static final int DEFAULT_GRID_HEIGHT = 3;

    /** Default depth (Z axis, blocks) of the local block grid sampled around the player. */
    public static final int DEFAULT_GRID_DEPTH = 7;

    /** Smallest legal value for any block grid dimension (a grid must exist). */
    public static final int MIN_GRID_DIMENSION = 1;

    /** Largest legal value for any block grid dimension, to keep per-tick payload size sane. */
    public static final int MAX_GRID_DIMENSION = 31;

    /** Default "log every Nth tick" divisor. 1 means every tick (~20Hz). */
    public static final int DEFAULT_SAMPLE_RATE_DIVISOR = 1;

    /** Smallest legal sample rate divisor. */
    public static final int MIN_SAMPLE_RATE_DIVISOR = 1;

    /** Default session log output directory, relative to the Minecraft run directory. */
    public static final String DEFAULT_LOG_OUTPUT_DIRECTORY = "herbert_logs";

    /**
     * Default world Y coordinate at/below which an AIR cell in the block grid is reclassified
     * as {@link dev.herbert.bridgelogger.model.BlockCategory#VOID} rather than plain AIR.
     *
     * <p>Hypixel Bridge maps do not use a dedicated "void" block; the void is simply open air
     * beneath the playable islands. This threshold is therefore a best-effort heuristic and is
     * exposed as {@code voidThresholdY} in the config so the community can tune it per map.</p>
     */
    public static final int DEFAULT_VOID_THRESHOLD_Y = 0;

    /** Maximum distance (blocks) at which another player is considered a trackable "opponent". */
    public static final double OPPONENT_MAX_DISTANCE_BLOCKS = 48.0;

    /** Connect timeout for all outbound HTTP calls (pastes.dev, Discord webhook), in milliseconds. */
    public static final int HTTP_CONNECT_TIMEOUT_MS = 10_000;

    /** Read timeout for all outbound HTTP calls, in milliseconds. */
    public static final int HTTP_READ_TIMEOUT_MS = 15_000;

    /** pastes.dev anonymous paste creation endpoint. */
    public static final String PASTES_DEV_POST_URL = "https://api.pastes.dev/post";

    /** Prefix used to build a human-viewable pastes.dev URL from a paste key. */
    public static final String PASTES_DEV_VIEW_URL_PREFIX = "https://pastes.dev/";

    /** MIME type used when uploading the session log to pastes.dev. */
    public static final String PASTES_DEV_CONTENT_TYPE = "text/plain";

    /** Discord embed side-bar color (decimal form of hex {@code 0x2ECC71}, a calm green). */
    public static final int DISCORD_EMBED_COLOR = 0x2ECC71;

    /** While idle, how often (in ticks) the Bridge-duel auto-detection heuristic is evaluated. */
    public static final int BRIDGE_DETECTION_CHECK_INTERVAL_TICKS = VANILLA_TICKS_PER_SECOND;

    /**
     * While recording, how many consecutive "no longer looks like a Bridge duel" ticks must
     * elapse before auto-stopping the session. Debounces transient scoreboard glitches/redraws.
     */
    public static final int BRIDGE_LOSS_DEBOUNCE_TICKS = 5 * VANILLA_TICKS_PER_SECOND;

    /** Capacity of the in-memory queue used by {@link dev.herbert.bridgelogger.serialize.AsyncFileWriter}. */
    public static final int ASYNC_WRITER_QUEUE_CAPACITY = 4096;

    /** How long {@link dev.herbert.bridgelogger.serialize.AsyncFileWriter} waits (ms) for its worker thread to stop on shutdown. */
    public static final long ASYNC_WRITER_SHUTDOWN_TIMEOUT_MS = 5_000L;

    /** Default substrings matched (case-insensitive) against the scoreboard sidebar title to detect a Bridge duel. */
    public static final String[] DEFAULT_BRIDGE_SCOREBOARD_TITLE_MATCHES = {"BRIDGE", "THE BRIDGE", "DUEL"};

    /** Default substrings matched (case-insensitive) against the current server address to detect Hypixel. */
    public static final String[] DEFAULT_BRIDGE_SERVER_ADDRESS_MATCHES = {"hypixel.net"};

    /**
     * Default number of seconds the session-end "display your username publicly?" chat prompt
     * (see {@link dev.herbert.bridgelogger.session.SessionManager}) waits for the player's
     * response before defaulting to a hashed-only username and proceeding with the upload.
     */
    public static final int DEFAULT_USERNAME_PROMPT_TIMEOUT_SECONDS = 30;

    /** Smallest legal value for the username-display prompt timeout, in seconds. */
    public static final int MIN_USERNAME_PROMPT_TIMEOUT_SECONDS = 1;

    /**
     * Discord's file-attachment size limit for a (non-boosted) webhook message, in bytes.
     * Documentation-only constant: {@link #DEFAULT_TARGET_CHUNK_SIZE_BYTES} is deliberately kept
     * below this with headroom for the surrounding multipart envelope (boundary markers, the
     * {@code payload_json} field, HTTP header overhead), not equal to it.
     */
    public static final long DISCORD_WEBHOOK_FILE_SIZE_LIMIT_BYTES = 8L * 1024 * 1024;

    /**
     * Default target size (bytes) for each split chunk when a session JSONL file is too large to
     * upload as a single Discord webhook file attachment (see
     * {@link dev.herbert.bridgelogger.serialize.SessionChunker}). Also doubles as the threshold
     * that decides whether a session needs chunking at all: a file at or under this size uploads
     * via the existing single-file pastes.dev + webhook-notify flow, unchanged; a file over it is
     * split into chunks each targeting this size. 7 MiB, leaving roughly 1 MiB of headroom below
     * {@link #DISCORD_WEBHOOK_FILE_SIZE_LIMIT_BYTES}.
     */
    public static final int DEFAULT_TARGET_CHUNK_SIZE_BYTES = 7 * 1024 * 1024;

    /** Smallest legal target chunk size, to keep the number of chunks/webhook requests sane. */
    public static final int MIN_TARGET_CHUNK_SIZE_BYTES = 64 * 1024;

    /**
     * Default delay (milliseconds) between successive chunk uploads to the same Discord webhook,
     * to stay well under Discord's documented rate limit of 30 requests per minute per webhook
     * (a 1000ms delay caps this mod's own chunk uploads at 60/minute in the worst case of
     * zero-latency requests, but leaves headroom against Discord's own request latency and any
     * other traffic sharing the same webhook).
     */
    public static final int DEFAULT_CHUNK_UPLOAD_DELAY_MILLIS = 1000;

    /** Smallest legal inter-chunk upload delay, in milliseconds (0 = no delay). */
    public static final int MIN_CHUNK_UPLOAD_DELAY_MILLIS = 0;

    /** Name of the Forge config category holding every BridgeLogger setting. */
    public static final String CONFIG_CATEGORY_GENERAL = "general";

    /** Root chat command name registered by {@link dev.herbert.bridgelogger.command.HerbertCommands}. */
    public static final String COMMAND_NAME = "herbert";

    private HerbertConstants() {
        // Static constant holder; never instantiated.
    }
}
