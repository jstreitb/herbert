// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.config;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import dev.herbert.bridgelogger.util.HerbertConstants;
import net.minecraftforge.common.config.Configuration;
import net.minecraftforge.common.config.Property;

/**
 * Thin, typed wrapper around Forge's {@link Configuration} system holding every user-facing
 * BridgeLogger setting.
 *
 * <p>All values are read once during {@link #load(File)} and cached as plain fields for cheap
 * access from the tick handler; call {@link #load(File)} again (e.g. in response to a
 * {@code /reload} style trigger) to pick up on-disk edits.</p>
 */
public final class HerbertConfig {

    private int blockGridWidth = HerbertConstants.DEFAULT_GRID_WIDTH;
    private int blockGridHeight = HerbertConstants.DEFAULT_GRID_HEIGHT;
    private int blockGridDepth = HerbertConstants.DEFAULT_GRID_DEPTH;
    private String logOutputDirectory = HerbertConstants.DEFAULT_LOG_OUTPUT_DIRECTORY;
    private int sampleRateDivisor = HerbertConstants.DEFAULT_SAMPLE_RATE_DIVISOR;
    private boolean opponentTrackingEnabled = true;
    private boolean scoreboardParsingEnabled = true;
    private String discordWebhookUrl = "";
    private boolean dryRunMode = false;
    private int voidThresholdY = HerbertConstants.DEFAULT_VOID_THRESHOLD_Y;
    private String[] bridgeScoreboardTitleMatches = HerbertConstants.DEFAULT_BRIDGE_SCOREBOARD_TITLE_MATCHES;
    private String[] bridgeServerAddressMatches = HerbertConstants.DEFAULT_BRIDGE_SERVER_ADDRESS_MATCHES;
    private int usernamePromptTimeoutSeconds = HerbertConstants.DEFAULT_USERNAME_PROMPT_TIMEOUT_SECONDS;
    private int targetChunkSizeBytes = HerbertConstants.DEFAULT_TARGET_CHUNK_SIZE_BYTES;
    private int chunkUploadDelayMillis = HerbertConstants.DEFAULT_CHUNK_UPLOAD_DELAY_MILLIS;

    private Configuration configuration;

    /**
     * Loads (creating with defaults if necessary) the BridgeLogger configuration from the given
     * file, populating every field on this wrapper. Safe to call multiple times.
     *
     * @param configFile the {@code .cfg} file Forge should read/write; must not be {@code null}
     */
    public void load(File configFile) {
        configuration = new Configuration(configFile);
        try {
            configuration.load();

            blockGridWidth = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "blockGridWidth",
                    HerbertConstants.DEFAULT_GRID_WIDTH, "Width (X axis, blocks) of the local block grid sampled around the player.");
            blockGridHeight = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "blockGridHeight",
                    HerbertConstants.DEFAULT_GRID_HEIGHT, "Height (Y axis, blocks) of the local block grid sampled around the player.");
            blockGridDepth = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "blockGridDepth",
                    HerbertConstants.DEFAULT_GRID_DEPTH, "Depth (Z axis, blocks) of the local block grid sampled around the player.");

            logOutputDirectory = getString(HerbertConstants.CONFIG_CATEGORY_GENERAL, "logOutputDirectory",
                    HerbertConstants.DEFAULT_LOG_OUTPUT_DIRECTORY,
                    "Directory (relative to the .minecraft run dir, or absolute) that session JSONL files are written to.");

            sampleRateDivisor = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "sampleRateDivisor",
                    HerbertConstants.DEFAULT_SAMPLE_RATE_DIVISOR,
                    "Log every Nth client tick. 1 = every tick (~20Hz), 2 = ~10Hz, etc.");

            opponentTrackingEnabled = getBoolean(HerbertConstants.CONFIG_CATEGORY_GENERAL, "opponentTrackingEnabled", true,
                    "If true, attempt to locate and record the nearest opponent's relative state each tick.");

            scoreboardParsingEnabled = getBoolean(HerbertConstants.CONFIG_CATEGORY_GENERAL, "scoreboardParsingEnabled", true,
                    "If true, attempt best-effort parsing of the Hypixel scoreboard for match context (scores, timer, kit).");

            // Try to read injected webhook URL from the bundled resource file (set at build time).
            String injectedWebhookDefault = loadInjectedWebhookUrl();
            discordWebhookUrl = getString(HerbertConstants.CONFIG_CATEGORY_GENERAL, "discordWebhookUrl", injectedWebhookDefault,
                    "Discord webhook URL that completed sessions are uploaded to directly (as a file attachment -- "
                            + "this mod does not use any third-party paste-hosting service). "
                            + "If left empty, the upload is skipped and a chat warning is shown instead. "
                            + "(This default may have been pre-filled at build time from webhook.txt if present.)");

            dryRunMode = getBoolean(HerbertConstants.CONFIG_CATEGORY_GENERAL, "dryRunMode", false,
                    "If true, sessions are still written to disk as JSONL but the Discord upload step is skipped entirely. Useful for local-only testing.");

            voidThresholdY = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "voidThresholdY",
                    HerbertConstants.DEFAULT_VOID_THRESHOLD_Y,
                    "World Y coordinate at/below which an air block-grid cell is classified as VOID instead of AIR. "
                            + "Hypixel does not expose a real void block, so this is a best-effort heuristic that may need per-map tuning.");

            bridgeScoreboardTitleMatches = getStringList(HerbertConstants.CONFIG_CATEGORY_GENERAL, "bridgeScoreboardTitleMatches",
                    HerbertConstants.DEFAULT_BRIDGE_SCOREBOARD_TITLE_MATCHES,
                    "Case-insensitive substrings checked against the scoreboard sidebar title to auto-detect a Bridge duel. "
                            + "Hypixel's exact wording may change; tune this list if auto-detection stops working.");

            bridgeServerAddressMatches = getStringList(HerbertConstants.CONFIG_CATEGORY_GENERAL, "bridgeServerAddressMatches",
                    HerbertConstants.DEFAULT_BRIDGE_SERVER_ADDRESS_MATCHES,
                    "Case-insensitive substrings checked against the current server address to auto-detect a Hypixel connection.");

            usernamePromptTimeoutSeconds = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "usernamePromptTimeoutSeconds",
                    HerbertConstants.DEFAULT_USERNAME_PROMPT_TIMEOUT_SECONDS,
                    "How many seconds the session-end \"display your username publicly?\" chat prompt waits for a "
                            + "response before defaulting to a hashed-only username and proceeding with the upload.");

            targetChunkSizeBytes = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "targetChunkSizeBytes",
                    HerbertConstants.DEFAULT_TARGET_CHUNK_SIZE_BYTES,
                    "Target size, in bytes, of each split chunk when a session JSONL file is too large to upload as a "
                            + "single Discord webhook file attachment (default 7 MiB, under Discord's 8 MiB webhook file "
                            + "limit). Also the threshold that decides whether a session needs chunking at all -- a file "
                            + "at or under this size uploads as a single file, unchanged.");

            chunkUploadDelayMillis = getInt(HerbertConstants.CONFIG_CATEGORY_GENERAL, "chunkUploadDelayMillis",
                    HerbertConstants.DEFAULT_CHUNK_UPLOAD_DELAY_MILLIS,
                    "Delay, in milliseconds, between successive chunk uploads to the Discord webhook when a session "
                            + "was split (see targetChunkSizeBytes), to stay under Discord's ~30-requests-per-minute "
                            + "per-webhook rate limit.");

            blockGridWidth = clamp(blockGridWidth, HerbertConstants.MIN_GRID_DIMENSION, HerbertConstants.MAX_GRID_DIMENSION);
            blockGridHeight = clamp(blockGridHeight, HerbertConstants.MIN_GRID_DIMENSION, HerbertConstants.MAX_GRID_DIMENSION);
            blockGridDepth = clamp(blockGridDepth, HerbertConstants.MIN_GRID_DIMENSION, HerbertConstants.MAX_GRID_DIMENSION);
            sampleRateDivisor = Math.max(sampleRateDivisor, HerbertConstants.MIN_SAMPLE_RATE_DIVISOR);
            usernamePromptTimeoutSeconds = Math.max(usernamePromptTimeoutSeconds, HerbertConstants.MIN_USERNAME_PROMPT_TIMEOUT_SECONDS);
            targetChunkSizeBytes = Math.max(targetChunkSizeBytes, HerbertConstants.MIN_TARGET_CHUNK_SIZE_BYTES);
            chunkUploadDelayMillis = Math.max(chunkUploadDelayMillis, HerbertConstants.MIN_CHUNK_UPLOAD_DELAY_MILLIS);
        } finally {
            if (configuration.hasChanged()) {
                configuration.save();
            }
        }
    }

    private int getInt(String category, String key, int defaultValue, String comment) {
        Property property = configuration.get(category, key, defaultValue, comment);
        return property.getInt(defaultValue);
    }

    private boolean getBoolean(String category, String key, boolean defaultValue, String comment) {
        Property property = configuration.get(category, key, defaultValue, comment);
        return property.getBoolean(defaultValue);
    }

    private String getString(String category, String key, String defaultValue, String comment) {
        Property property = configuration.get(category, key, defaultValue, comment);
        return property.getString();
    }

    private String[] getStringList(String category, String key, String[] defaultValue, String comment) {
        Property property = configuration.get(category, key, defaultValue, comment);
        String[] values = property.getStringList();
        return values == null ? defaultValue : values;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    /** @return configured block grid width (X axis, blocks), already clamped to a sane range */
    public int getBlockGridWidth() {
        return blockGridWidth;
    }

    /** @return configured block grid height (Y axis, blocks), already clamped to a sane range */
    public int getBlockGridHeight() {
        return blockGridHeight;
    }

    /** @return configured block grid depth (Z axis, blocks), already clamped to a sane range */
    public int getBlockGridDepth() {
        return blockGridDepth;
    }

    /** @return configured session log output directory, relative to the run dir or absolute */
    public String getLogOutputDirectory() {
        return logOutputDirectory;
    }

    /** @return "log every Nth tick" divisor, always {@code >= 1} */
    public int getSampleRateDivisor() {
        return sampleRateDivisor;
    }

    /** @return whether opponent tracking is enabled */
    public boolean isOpponentTrackingEnabled() {
        return opponentTrackingEnabled;
    }

    /** @return whether best-effort scoreboard parsing is enabled */
    public boolean isScoreboardParsingEnabled() {
        return scoreboardParsingEnabled;
    }

    /** @return configured Discord webhook URL, or an empty string if not configured */
    public String getDiscordWebhookUrl() {
        return discordWebhookUrl;
    }

    /** @return whether dry-run mode (write-to-disk only, no upload) is enabled */
    public boolean isDryRunMode() {
        return dryRunMode;
    }

    /** @return the void-threshold Y coordinate used by {@link dev.herbert.bridgelogger.capture.BlockGridMapper} */
    public int getVoidThresholdY() {
        return voidThresholdY;
    }

    /** @return case-insensitive substrings matched against the scoreboard title to detect a Bridge duel */
    public String[] getBridgeScoreboardTitleMatches() {
        return bridgeScoreboardTitleMatches;
    }

    /** @return case-insensitive substrings matched against the server address to detect Hypixel */
    public String[] getBridgeServerAddressMatches() {
        return bridgeServerAddressMatches;
    }

    /** @return seconds the session-end username-display chat prompt waits before defaulting, always {@code >= 1} */
    public int getUsernamePromptTimeoutSeconds() {
        return usernamePromptTimeoutSeconds;
    }

    /**
     * @return target chunk size in bytes (also the single-file-vs-chunked threshold), always
     *         {@code >= } {@link HerbertConstants#MIN_TARGET_CHUNK_SIZE_BYTES}
     */
    public int getTargetChunkSizeBytes() {
        return targetChunkSizeBytes;
    }

    /** @return delay in milliseconds between successive chunk uploads, always {@code >= 0} */
    public int getChunkUploadDelayMillis() {
        return chunkUploadDelayMillis;
    }

    /**
     * Attempts to load the webhook URL from the bundled webhook.properties resource file
     * (injected at build time from webhook.txt, if present). This provides a convenient
     * "out of box" default for community servers that pre-build the mod with a specific
     * webhook URL.
     *
     * @return the webhook URL from webhook.properties, or an empty string if not found/readable.
     */
    private String loadInjectedWebhookUrl() {
        InputStream stream = HerbertConfig.class.getResourceAsStream("/webhook.properties");
        if (stream == null) {
            // Not a build that injected a webhook.txt -- this is the normal case for a
            // from-source build with no pre-configured webhook, not an error.
            return "";
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("discordWebhookUrl=") && !line.startsWith("#")) {
                    String value = line.substring("discordWebhookUrl=".length()).trim();
                    if (!value.isEmpty()) {
                        return value;
                    }
                }
            }
        } catch (IOException e) {
            // webhook.properties exists but could not be read; fall back to the empty default
            // rather than failing mod initialization over an optional, best-effort convenience
            // file. The user can still set discordWebhookUrl manually in the generated config.
        }
        return "";
    }
}
