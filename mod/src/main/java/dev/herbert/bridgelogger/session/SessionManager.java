// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.session;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.logging.log4j.Level;

import dev.herbert.bridgelogger.capture.BridgeDetector;
import dev.herbert.bridgelogger.capture.TickSampler;
import dev.herbert.bridgelogger.config.HerbertConfig;
import dev.herbert.bridgelogger.model.SessionHeader;
import dev.herbert.bridgelogger.model.TickSnapshot;
import dev.herbert.bridgelogger.serialize.AsyncFileWriter;
import dev.herbert.bridgelogger.serialize.SessionChunk;
import dev.herbert.bridgelogger.serialize.SessionChunker;
import dev.herbert.bridgelogger.serialize.SessionSerializer;
import dev.herbert.bridgelogger.upload.DiscordWebhookNotifier;
import dev.herbert.bridgelogger.util.HashUtil;
import dev.herbert.bridgelogger.util.HerbertConstants;
import dev.herbert.bridgelogger.util.ScoreboardUtil;
import net.minecraft.block.Block;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.gui.GuiChat;
import net.minecraft.client.multiplayer.ServerData;
import net.minecraft.init.Blocks;
import net.minecraft.item.ItemStack;
import net.minecraft.util.BlockPos;
import net.minecraft.util.ChatComponentText;
import net.minecraft.world.World;
import net.minecraftforge.client.event.GuiOpenEvent;
import net.minecraftforge.event.entity.player.AttackEntityEvent;
import net.minecraftforge.event.entity.player.PlayerInteractEvent;
import net.minecraftforge.fml.common.FMLLog;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import net.minecraftforge.fml.common.network.FMLNetworkEvent;

/**
 * Owns the BridgeLogger recording session state machine: auto-detecting the start/end of a
 * Hypixel Bridge duel, driving per-tick sampling into the current session file, and kicking off
 * the (fully asynchronous) upload flow when a session ends.
 *
 * <p>This class is the single Forge event subscriber for everything session-lifecycle related.
 * Every event handler is wrapped in a try/catch so that no unexpected condition (malformed
 * scoreboard, disconnected world, etc.) can ever propagate an exception out of the client tick
 * loop and disrupt the player's game.</p>
 */
public final class SessionManager {

    /** Recording state of the session state machine. */
    private enum SessionState {
        IDLE,
        RECORDING
    }

    private final HerbertConfig config;
    private final TickSampler tickSampler;
    private final SessionSerializer serializer;
    private final DiscordWebhookNotifier discordWebhookNotifier;
    private final ExecutorService uploadExecutor;
    private final ScheduledExecutorService promptTimeoutExecutor;

    private volatile SessionState state = SessionState.IDLE;

    private AsyncFileWriter writer;
    private File currentLogFile;
    private SessionHeader currentHeader;
    private String currentPlayerUsername;
    private String sessionId;
    private long sessionStartMillis;
    private long tickCounter;
    private long rawTickCounter;
    private int idleDetectionCounter;
    private int bridgeMissCounter;

    /**
     * Whether a session-end "display your username publicly?" prompt is currently awaiting the
     * player's response. Read from the {@code GuiOpenEvent} handler (client thread) and from the
     * prompt-timeout callback (background executor thread, always hopped back onto the client
     * thread before touching any other state -- see {@link #onUsernamePromptTimeout()}); every
     * write to this field, and every method that reads it to decide on a state transition, is
     * guarded by this instance's monitor (see the {@code synchronized} prompt methods below).
     */
    private volatile boolean awaitingUsernamePrompt = false;

    private int usernamePromptInvalidAttempts;
    private ScheduledFuture<?> promptTimeoutFuture;
    private UsernamePromptCallback usernamePromptCallback;

    /**
     * Creates a session manager bound to the given (live-updating) config.
     *
     * @param config the mod configuration; must not be {@code null}
     */
    public SessionManager(HerbertConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        this.config = config;
        this.tickSampler = new TickSampler(config);
        this.serializer = new SessionSerializer();
        this.discordWebhookNotifier = new DiscordWebhookNotifier();
        this.uploadExecutor = Executors.newSingleThreadExecutor(new DaemonThreadFactory("Herbert-Upload"));
        this.promptTimeoutExecutor = Executors.newSingleThreadScheduledExecutor(new DaemonThreadFactory("Herbert-PromptTimeout"));
    }

    /**
     * Main client tick handler. Drives auto-detection while idle and sampling/auto-stop while
     * recording. Registered on the game (Forge) event bus by the mod's main class.
     *
     * @param event the tick event; only {@link TickEvent.Phase#END} client ticks are processed
     */
    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        if (event.phase != TickEvent.Phase.END) {
            return;
        }
        try {
            Minecraft mc = Minecraft.getMinecraft();
            EntityPlayerSP player = mc.thePlayer;
            World world = mc.theWorld;

            if (player == null || world == null) {
                if (state == SessionState.RECORDING) {
                    stopSession("local player or world became unavailable");
                }
                return;
            }

            if (state == SessionState.IDLE) {
                handleIdleTick(mc, player, world);
            } else {
                handleRecordingTick(mc, player, world);
            }
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Unexpected error in BridgeLogger tick handler");
        }
    }

    /**
     * Records a left-click/attack event for inclusion in the next sampled tick, when recording.
     *
     * @param event the Forge attack event
     */
    @SubscribeEvent
    public void onAttackEntity(AttackEntityEvent event) {
        if (state != SessionState.RECORDING) {
            return;
        }
        try {
            String targetType = (event.target != null) ? event.target.getClass().getSimpleName() : null;
            tickSampler.recordAttack(targetType);
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Error handling attack event");
        }
    }

    /**
     * Records a right-click block-place event for inclusion in the next sampled tick, when
     * recording. Only fires for right-clicks on a block while holding a placeable block item;
     * this is a best-effort approximation since Forge's interact event does not guarantee vanilla
     * ultimately accepted the placement (e.g. it could be blocked by another block already there).
     *
     * @param event the Forge player-interact event
     */
    @SubscribeEvent
    public void onPlayerInteract(PlayerInteractEvent event) {
        if (state != SessionState.RECORDING) {
            return;
        }
        try {
            if (event.action != PlayerInteractEvent.Action.RIGHT_CLICK_BLOCK) {
                return;
            }
            ItemStack heldStack = event.entityPlayer.getHeldItem();
            if (heldStack == null || heldStack.getItem() == null) {
                return;
            }
            Block block = Block.getBlockFromItem(heldStack.getItem());
            if (block == null || block == Blocks.air) {
                // Not a placeable block item; nothing was placed.
                return;
            }
            String blockType = registryName(block);
            BlockPos placementPos = (event.pos != null && event.face != null) ? event.pos.offset(event.face) : null;
            tickSampler.recordPlace(blockType, placementPos);
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Error handling player interact event");
        }
    }

    /**
     * Auto-stops any active session when the client disconnects from the server, since there is
     * no further game state to sample.
     *
     * @param event the FML network disconnection event
     */
    @SubscribeEvent
    public void onDisconnect(FMLNetworkEvent.ClientDisconnectionFromServerEvent event) {
        try {
            if (state == SessionState.RECORDING) {
                stopSession("disconnected from server");
            }
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Error handling disconnect event");
        }
    }

    /**
     * Intercepts the vanilla chat GUI opening while a session-end username-display prompt is
     * awaiting a response, replacing it with {@link HerbertPromptChatGui} so the player's next
     * "y"/"n" answer is captured client-side instead of being sent to the server as a public chat
     * message. A no-op outside the narrow window a prompt is actually pending, and a no-op for
     * any GUI other than the chat box (inventory, pause menu, etc. all open normally).
     *
     * @param event the Forge GUI-open event, whose {@code gui} field this replaces when applicable
     */
    @SubscribeEvent
    public void onGuiOpen(GuiOpenEvent event) {
        if (!awaitingUsernamePrompt) {
            return;
        }
        try {
            if (event.gui instanceof GuiChat && !(event.gui instanceof HerbertPromptChatGui)) {
                event.gui = new HerbertPromptChatGui(this);
            }
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Error intercepting chat GUI for username-display prompt");
        }
    }

    private void handleIdleTick(Minecraft mc, EntityPlayerSP player, World world) {
        idleDetectionCounter++;
        if (idleDetectionCounter < HerbertConstants.BRIDGE_DETECTION_CHECK_INTERVAL_TICKS) {
            return;
        }
        idleDetectionCounter = 0;

        String serverAddress = getServerAddress(mc);
        String scoreboardTitle = ScoreboardUtil.getSidebarTitle(world);
        boolean detected = BridgeDetector.looksLikeBridgeDuel(serverAddress, scoreboardTitle,
                config.getBridgeServerAddressMatches(), config.getBridgeScoreboardTitleMatches());
        if (detected) {
            startSession(true, player);
        }
    }

    private void handleRecordingTick(Minecraft mc, EntityPlayerSP player, World world) {
        rawTickCounter++;
        if (rawTickCounter % Math.max(1, config.getSampleRateDivisor()) == 0) {
            sampleAndWrite(player, world);
        }

        String serverAddress = getServerAddress(mc);
        String scoreboardTitle = ScoreboardUtil.getSidebarTitle(world);
        boolean stillDetected = BridgeDetector.looksLikeBridgeDuel(serverAddress, scoreboardTitle,
                config.getBridgeServerAddressMatches(), config.getBridgeScoreboardTitleMatches());
        if (stillDetected) {
            bridgeMissCounter = 0;
        } else {
            bridgeMissCounter++;
            if (bridgeMissCounter >= HerbertConstants.BRIDGE_LOSS_DEBOUNCE_TICKS) {
                stopSession("Bridge duel signals no longer detected");
            }
        }
    }

    private void sampleAndWrite(EntityPlayerSP player, World world) {
        try {
            TickSnapshot snapshot = tickSampler.sample(tickCounter, player, world);
            tickCounter++;
            String line = serializer.serializeTick(snapshot);
            if (line != null && writer != null) {
                writer.enqueue(line);
            }
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Failed to sample/write tick %d", tickCounter);
        }
    }

    /**
     * Starts a new recording session (manual override entry point). No-op with a chat message
     * if already recording, or if the player is not currently in-game.
     */
    public synchronized void startManual() {
        Minecraft mc = Minecraft.getMinecraft();
        EntityPlayerSP player = mc.thePlayer;
        if (player == null) {
            chat("Herbert: cannot start recording, you are not in-game.");
            return;
        }
        if (state == SessionState.RECORDING) {
            chat("Herbert: already recording (session " + sessionId + ").");
            return;
        }
        startSession(false, player);
    }

    /**
     * Stops the current recording session (manual override entry point), triggering the upload
     * flow. No-op with a chat message if not currently recording.
     */
    public synchronized void stopManual() {
        if (state != SessionState.RECORDING) {
            chat("Herbert: not currently recording.");
            return;
        }
        stopSession("manual /herbert stop");
    }

    /**
     * @return a human-readable status line describing the current session state, for the
     *         {@code /herbert status} command
     */
    public synchronized String getStatusText() {
        if (state == SessionState.IDLE) {
            return "Herbert: idle (not recording).";
        }
        long elapsedSeconds = (System.currentTimeMillis() - sessionStartMillis) / 1000L;
        return String.format("Herbert: recording session %s — %d ticks captured, %ds elapsed.",
                sessionId, tickCounter, elapsedSeconds);
    }

    private synchronized void startSession(boolean auto, EntityPlayerSP player) {
        if (state == SessionState.RECORDING) {
            return;
        }
        try {
            String newSessionId = UUID.randomUUID().toString();
            String startTimestamp = Instant.now().toString();
            String username = player.getName();
            String usernameHash = HashUtil.sha256Hex(username);
            // playerUsernameDisplay is unknown until the session ends and the player has answered
            // the display-name prompt (see stopSession/beginUsernameDisplayPrompt); the header
            // written to disk here is later rewritten in place if they opt in -- see
            // finalizeSessionHeaderWithDisplayName.
            SessionHeader header = new SessionHeader(HerbertConstants.SCHEMA_VERSION, HerbertConstants.MOD_VERSION,
                    newSessionId, startTimestamp, usernameHash, null);

            File logDir = resolveLogDirectory();
            File logFile = new File(logDir, newSessionId + ".jsonl");

            AsyncFileWriter newWriter = new AsyncFileWriter();
            newWriter.open(logFile);

            String headerLine = serializer.serializeHeader(header);
            if (headerLine != null) {
                newWriter.enqueue(headerLine);
            }

            this.writer = newWriter;
            this.currentLogFile = logFile;
            this.currentHeader = header;
            this.currentPlayerUsername = username;
            this.sessionId = newSessionId;
            this.sessionStartMillis = System.currentTimeMillis();
            this.tickCounter = 0L;
            this.rawTickCounter = 0L;
            this.bridgeMissCounter = 0;
            this.tickSampler.reset();
            this.state = SessionState.RECORDING;

            chat("Herbert: recording started" + (auto ? " (auto-detected Bridge duel)" : "") + ". Session " + newSessionId);
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Failed to start recording session");
            chat("Herbert: failed to start recording (" + e.getMessage() + ").");
        }
    }

    private synchronized void stopSession(String reason) {
        if (state != SessionState.RECORDING) {
            return;
        }
        state = SessionState.IDLE;

        final long durationSeconds = (System.currentTimeMillis() - sessionStartMillis) / 1000L;
        final long finalTickCount = tickCounter;
        final File fileToUpload = currentLogFile;
        final AsyncFileWriter writerToClose = writer;
        final SessionHeader headerToFinalize = currentHeader;
        final String rawUsernameForHeader = currentPlayerUsername;
        writer = null;
        currentLogFile = null;
        currentHeader = null;
        currentPlayerUsername = null;

        FMLLog.log(HerbertConstants.MOD_NAME, Level.INFO, "Session %s stopped (%s), %d ticks over %ds",
                sessionId, reason, finalTickCount, durationSeconds);

        // Upload must wait for the player's answer to the display-name prompt (see class javadoc
        // on awaitingUsernamePrompt) before the file's header is finalized and read for upload.
        beginUsernameDisplayPrompt(new UsernamePromptCallback() {
            @Override
            public void onResolved(final boolean includeDisplayName) {
                uploadExecutor.submit(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            writerToClose.close();
                        } catch (IOException e) {
                            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Failed to close session log file");
                        }
                        if (includeDisplayName) {
                            finalizeSessionHeaderWithDisplayName(fileToUpload, headerToFinalize, rawUsernameForHeader);
                        }
                        performUploadFlow(fileToUpload, headerToFinalize.sessionId, durationSeconds, finalTickCount);
                    }
                });
            }
        });
    }

    /**
     * Callback invoked exactly once, from {@link #resolvePromptLocked(boolean)}, once the
     * session-end username-display prompt has been answered (by the player, by the two-invalid-
     * inputs default, or by timeout).
     */
    private interface UsernamePromptCallback {
        /**
         * @param includeDisplayName {@code true} if the player opted in to a plaintext
         *        {@code player_username_display} header field; {@code false} for the hash-only
         *        default
         */
        void onResolved(boolean includeDisplayName);
    }

    /**
     * Prints the session-end "display your username publicly?" prompt and arms the response
     * capture/timeout machinery, invoking {@code callback} exactly once when resolved (by a valid
     * "y"/"n" answer via {@link #onUsernamePromptChatCaptured(String)}, by defaulting after two
     * invalid answers, or by {@link #onUsernamePromptTimeout()}).
     *
     * <p>Must only be called from the client thread (every realistic caller -- the client tick
     * handler, the disconnect event, and the {@code /herbert stop} command -- already is), since
     * it calls {@link #chat(String)} directly.</p>
     *
     * @param callback notified with the final display-name decision; must not be {@code null}
     */
    private synchronized void beginUsernameDisplayPrompt(UsernamePromptCallback callback) {
        if (awaitingUsernamePrompt) {
            // Defensive: an earlier prompt (from a session stopped and restarted faster than its
            // own timeout) is still pending. Resolve it as "n" immediately rather than let two
            // prompts fight over the GuiOpenEvent hook -- see the onGuiOpen javadoc.
            UsernamePromptCallback stalePromptCallback = usernamePromptCallback;
            cancelPromptTimeoutLocked();
            awaitingUsernamePrompt = false;
            usernamePromptCallback = null;
            if (stalePromptCallback != null) {
                stalePromptCallback.onResolved(false);
            }
        }

        usernamePromptInvalidAttempts = 0;
        usernamePromptCallback = callback;
        awaitingUsernamePrompt = true;
        int timeoutSeconds = Math.max(HerbertConstants.MIN_USERNAME_PROMPT_TIMEOUT_SECONDS, config.getUsernamePromptTimeoutSeconds());
        promptTimeoutFuture = promptTimeoutExecutor.schedule(new Runnable() {
            @Override
            public void run() {
                onUsernamePromptTimeout();
            }
        }, timeoutSeconds, TimeUnit.SECONDS);

        chat("Herbert: Session stopped. Display your username publicly? [y/n]");
    }

    /**
     * Handles the player's captured chat response to a pending username-display prompt (routed
     * here by {@link HerbertPromptChatGui#sendChatMessage(String)}, which runs on the client
     * thread during GUI input handling -- so {@link #chat(String)} is safe to call directly).
     * Accepts {@code "y"}/{@code "n"} case-insensitively; anything else re-prints the prompt once
     * and waits again, defaulting to hashed-only after a second invalid answer.
     *
     * @param rawMessage the trimmed, non-empty text the player typed; never sent to the server
     */
    synchronized void onUsernamePromptChatCaptured(String rawMessage) {
        if (!awaitingUsernamePrompt) {
            // Stale capture (e.g. the prompt already timed out on the exact same tick); ignore.
            return;
        }
        String normalized = rawMessage == null ? "" : rawMessage.trim().toLowerCase(Locale.ROOT);
        if ("y".equals(normalized)) {
            resolvePromptLocked(true);
        } else if ("n".equals(normalized)) {
            resolvePromptLocked(false);
        } else {
            usernamePromptInvalidAttempts++;
            if (usernamePromptInvalidAttempts >= 2) {
                chat("Herbert: Invalid input twice — defaulting to hashed username.");
                resolvePromptLocked(false);
            } else {
                chat("Herbert: Session stopped. Display your username publicly? [y/n]");
            }
        }
    }

    /**
     * Fires when {@link #promptTimeoutFuture} elapses with no valid response yet. Always hops
     * onto the client thread before touching any shared state or calling {@link #chat(String)},
     * since the scheduled-executor callback itself runs on a background thread.
     */
    private void onUsernamePromptTimeout() {
        Minecraft.getMinecraft().addScheduledTask(new Runnable() {
            @Override
            public void run() {
                synchronized (SessionManager.this) {
                    if (!awaitingUsernamePrompt) {
                        // Already resolved by a chat response that arrived just before the timeout.
                        return;
                    }
                    chat("Herbert: No response — defaulting to hashed username only.");
                    resolvePromptLocked(false);
                }
            }
        });
    }

    /**
     * Finalizes the pending username-display prompt: cancels its timeout, clears the awaiting
     * state, and invokes its callback. Must be called while holding this instance's monitor.
     *
     * @param includeDisplayName the resolved decision to pass to the pending callback
     */
    private void resolvePromptLocked(boolean includeDisplayName) {
        cancelPromptTimeoutLocked();
        awaitingUsernamePrompt = false;
        UsernamePromptCallback callback = usernamePromptCallback;
        usernamePromptCallback = null;
        if (callback != null) {
            chat("Herbert: Uploading...");
            callback.onResolved(includeDisplayName);
        }
    }

    /**
     * Cancels {@link #promptTimeoutFuture} if one is armed. Must be called while holding this
     * instance's monitor.
     */
    private void cancelPromptTimeoutLocked() {
        if (promptTimeoutFuture != null) {
            promptTimeoutFuture.cancel(false);
            promptTimeoutFuture = null;
        }
    }

    /**
     * Rewrites the completed session file's first (header) line in place to include the raw
     * {@code player_username_display} field, once the player has opted in. Always runs on the
     * upload executor's background thread, after {@link AsyncFileWriter#close()} has already
     * flushed and closed the file (never concurrently with the async writer). On any failure the
     * file is left exactly as originally written (hash-only header) and the failure is logged --
     * consistent with this mod's "a logging bug should never lose or corrupt already-captured
     * data" philosophy.
     *
     * @param file the completed session log file
     * @param originalHeader the header exactly as written to disk at session start
     * @param rawUsername the player's raw username to add to the header
     */
    private void finalizeSessionHeaderWithDisplayName(File file, SessionHeader originalHeader, String rawUsername) {
        try {
            SessionHeader finalHeader = new SessionHeader(originalHeader.schemaVersion, originalHeader.herbertModVersion,
                    originalHeader.sessionId, originalHeader.recordingStartTimestamp, originalHeader.playerUsernameHash,
                    rawUsername);
            String headerLine = serializer.serializeHeader(finalHeader);
            if (headerLine == null) {
                FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR,
                        "Failed to serialize finalized session header with display username for %s; header on disk stays hash-only.",
                        file.getAbsolutePath());
                return;
            }
            rewriteFirstLine(file, headerLine);
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e,
                    "Failed to rewrite session header with display username for %s; header on disk stays hash-only.",
                    file.getAbsolutePath());
        }
    }

    /**
     * Replaces the first line of {@code file} with {@code newFirstLine}, leaving every subsequent
     * line untouched. Used to finalize the session header after the fact (see
     * {@link #finalizeSessionHeaderWithDisplayName(File, SessionHeader, String)}), since the
     * player's display-name decision is only known after every tick line has already been
     * written.
     *
     * @param file the file to rewrite
     * @param newFirstLine the replacement first line, without a trailing newline
     * @throws IOException if reading or rewriting the file fails
     */
    private static void rewriteFirstLine(File file, String newFirstLine) throws IOException {
        List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
        StringBuilder rewritten = new StringBuilder();
        rewritten.append(newFirstLine).append(System.lineSeparator());
        for (int i = 1; i < lines.size(); i++) {
            rewritten.append(lines.get(i)).append(System.lineSeparator());
        }
        Files.write(file.toPath(), rewritten.toString().getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Executes the upload flow for a completed session. Always runs on the upload executor's
     * background thread, never on the client thread.
     *
     * <p>Every session is uploaded directly to the configured Discord webhook as a file
     * attachment -- this mod does not use any third-party paste-hosting service. Dispatches to
     * one of two paths depending on the file's size:</p>
     * <ul>
     *   <li>At or under {@code config.getTargetChunkSizeBytes()}: uploads the whole file as a
     *       single webhook message (see {@link DiscordWebhookNotifier#uploadSessionFile}).</li>
     *   <li>Over that size: {@link #performChunkedUploadFlow(File, String, String, long, long)}
     *       splits the file and uploads each chunk as its own webhook message. A session is
     *       <b>never</b> silently dropped for being too large.</li>
     * </ul>
     *
     * <p>On any failure, the local JSONL file is left intact on disk; the failure is logged and
     * reported to the player via chat -- and, since there is no longer a separate "paste
     * succeeded" step to falsely report as complete, a failed Discord upload is now always
     * reported as a failure rather than silently swallowed.</p>
     *
     * @param file the completed session JSONL file
     * @param sessionId the session's UUID (captured by the caller before this ran, rather than
     *        read from the {@code sessionId} instance field, since a new session may already have
     *        started and overwritten that field by the time this background task runs)
     * @param durationSeconds wall-clock duration of the recorded session, in seconds
     * @param tickCount number of ticks recorded in the session
     */
    private void performUploadFlow(File file, String sessionId, long durationSeconds, long tickCount) {
        if (config.isDryRunMode()) {
            scheduleChat("Herbert: dry-run mode enabled, upload skipped. Log saved at " + file.getAbsolutePath());
            return;
        }
        String webhookUrl = config.getDiscordWebhookUrl();
        if (webhookUrl == null || webhookUrl.isEmpty()) {
            scheduleChat("Herbert: no webhook configured, skipping upload");
            return;
        }

        if (file.length() > config.getTargetChunkSizeBytes()) {
            performChunkedUploadFlow(file, sessionId, webhookUrl, durationSeconds, tickCount);
            return;
        }

        try {
            discordWebhookNotifier.uploadSessionFile(webhookUrl, file, sessionId, durationSeconds, tickCount,
                    HerbertConstants.SCHEMA_VERSION, HerbertConstants.MOD_VERSION);
        } catch (Exception uploadError) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, uploadError, "Session upload failed for %s", file.getAbsolutePath());
            scheduleChat("Upload failed: " + uploadError.getMessage() + " (log kept at " + file.getAbsolutePath() + ")");
            return;
        }

        scheduleChat("Herbert: Upload complete.");
    }

    /**
     * Splits an oversized session file into chunks (see {@link SessionChunker}) and uploads them
     * to Discord one at a time, each as its own file-attachment webhook message, waiting
     * {@code config.getChunkUploadDelayMillis()} between uploads to respect Discord's per-webhook
     * rate limit. Always runs on the upload executor's background thread.
     *
     * <p>If splitting itself fails (e.g. disk full, permissions error), the original file is left
     * untouched and the failure is reported like any other upload failure. If an individual
     * chunk's upload fails, the sequence stops immediately: chunks already uploaded are not
     * retried (Discord already has them), and every chunk from the failing one onward remains on
     * disk in {@code file.getParentFile()} exactly as {@link SessionChunker} wrote it, so nothing
     * is lost.</p>
     *
     * @param file the completed, oversized session JSONL file
     * @param sessionId the session's UUID
     * @param webhookUrl the configured Discord webhook URL (already validated non-empty by the caller)
     * @param durationSeconds wall-clock duration of the recorded session, in seconds
     * @param tickCount number of ticks recorded in the session
     */
    private void performChunkedUploadFlow(File file, String sessionId, String webhookUrl, long durationSeconds, long tickCount) {
        List<SessionChunk> chunks;
        try {
            SessionChunker chunker = new SessionChunker(config.getTargetChunkSizeBytes());
            chunks = chunker.split(file, file.getParentFile(), sessionId);
        } catch (IOException e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e,
                    "Failed to split oversized session file %s into chunks", file.getAbsolutePath());
            scheduleChat("Herbert: failed to split oversized session file (" + e.getMessage()
                    + "); log kept at " + file.getAbsolutePath());
            return;
        }

        int total = chunks.size();
        scheduleChat(String.format(Locale.ROOT,
                "Herbert: Session too large for a single upload — splitting into %d parts. (uploading 1/%d...)", total, total));

        for (int i = 0; i < total; i++) {
            SessionChunk chunk = chunks.get(i);
            int partNumber = chunk.getPartNumber();
            try {
                discordWebhookNotifier.uploadChunk(webhookUrl, chunk, sessionId, durationSeconds, tickCount,
                        HerbertConstants.SCHEMA_VERSION, HerbertConstants.MOD_VERSION);
            } catch (Exception uploadError) {
                FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, uploadError,
                        "Chunk upload failed for part %d/%d of session %s", partNumber, total, sessionId);
                scheduleChat(String.format(Locale.ROOT, "Herbert: Upload stopped at part %d/%d. Parts %s saved locally at %s.",
                        partNumber, total, formatPartRange(partNumber, total), file.getParentFile().getAbsolutePath()));
                return;
            }

            scheduleChat(String.format(Locale.ROOT, "Herbert: Uploaded part %d/%d.", partNumber, total));

            boolean isLastChunk = (i == total - 1);
            if (!isLastChunk) {
                sleepUninterruptibly(config.getChunkUploadDelayMillis());
            }
        }

        scheduleChat(String.format(Locale.ROOT, "Herbert: All %d parts uploaded successfully. (%d ticks total, %ds)",
                total, tickCount, durationSeconds));
    }

    /**
     * @param fromPartNumber 1-based part number to start the range at (inclusive)
     * @param toPartNumber 1-based part number to end the range at (inclusive)
     * @return {@code "N"} if the range is a single part, else {@code "N-M"}
     */
    private static String formatPartRange(int fromPartNumber, int toPartNumber) {
        return fromPartNumber == toPartNumber ? String.valueOf(fromPartNumber) : fromPartNumber + "-" + toPartNumber;
    }

    /**
     * Sleeps the calling (background upload executor) thread for {@code millis}, restoring the
     * interrupt flag and returning early if interrupted rather than throwing -- an interrupted
     * inter-chunk delay should not be treated as an upload failure.
     */
    private static void sleepUninterruptibly(long millis) {
        if (millis <= 0) {
            return;
        }
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private File resolveLogDirectory() {
        Minecraft mc = Minecraft.getMinecraft();
        String configured = config.getLogOutputDirectory();
        File configuredFile = new File(configured);
        if (configuredFile.isAbsolute()) {
            return configuredFile;
        }
        return new File(mc.mcDataDir, configured);
    }

    private String getServerAddress(Minecraft mc) {
        try {
            ServerData data = mc.getCurrentServerData();
            return data != null ? data.serverIP : null;
        } catch (Exception e) {
            return null;
        }
    }

    private static String registryName(Block block) {
        try {
            Object name = Block.blockRegistry.getNameForObject(block);
            return name == null ? null : name.toString();
        } catch (Exception e) {
            return null;
        }
    }

    private void chat(String message) {
        Minecraft mc = Minecraft.getMinecraft();
        if (mc.thePlayer != null) {
            mc.thePlayer.addChatMessage(new ChatComponentText(message));
        }
    }

    private void scheduleChat(final String message) {
        final Minecraft mc = Minecraft.getMinecraft();
        mc.addScheduledTask(new Runnable() {
            @Override
            public void run() {
                chat(message);
            }
        });
    }

    /** Simple named-daemon-thread factory so the upload executor never prevents JVM shutdown. */
    private static final class DaemonThreadFactory implements ThreadFactory {
        private final String namePrefix;
        private final AtomicInteger counter = new AtomicInteger(1);

        DaemonThreadFactory(String namePrefix) {
            this.namePrefix = namePrefix;
        }

        @Override
        public Thread newThread(Runnable runnable) {
            Thread thread = new Thread(runnable, namePrefix + "-" + counter.getAndIncrement());
            thread.setDaemon(true);
            return thread;
        }
    }
}
