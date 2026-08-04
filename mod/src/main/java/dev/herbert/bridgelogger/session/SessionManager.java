package dev.herbert.bridgelogger.session;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.logging.log4j.Level;

import dev.herbert.bridgelogger.capture.BridgeDetector;
import dev.herbert.bridgelogger.capture.TickSampler;
import dev.herbert.bridgelogger.config.HerbertConfig;
import dev.herbert.bridgelogger.model.SessionHeader;
import dev.herbert.bridgelogger.model.TickSnapshot;
import dev.herbert.bridgelogger.serialize.AsyncFileWriter;
import dev.herbert.bridgelogger.serialize.SessionSerializer;
import dev.herbert.bridgelogger.upload.DiscordWebhookNotifier;
import dev.herbert.bridgelogger.upload.PastesDevUploader;
import dev.herbert.bridgelogger.upload.UploadResult;
import dev.herbert.bridgelogger.util.HashUtil;
import dev.herbert.bridgelogger.util.HerbertConstants;
import dev.herbert.bridgelogger.util.ScoreboardUtil;
import net.minecraft.block.Block;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.multiplayer.ServerData;
import net.minecraft.init.Blocks;
import net.minecraft.item.ItemStack;
import net.minecraft.util.BlockPos;
import net.minecraft.util.ChatComponentText;
import net.minecraft.world.World;
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
    private final PastesDevUploader pastesDevUploader;
    private final DiscordWebhookNotifier discordWebhookNotifier;
    private final ExecutorService uploadExecutor;

    private volatile SessionState state = SessionState.IDLE;

    private AsyncFileWriter writer;
    private File currentLogFile;
    private String sessionId;
    private long sessionStartMillis;
    private long tickCounter;
    private long rawTickCounter;
    private int idleDetectionCounter;
    private int bridgeMissCounter;

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
        this.pastesDevUploader = new PastesDevUploader();
        this.discordWebhookNotifier = new DiscordWebhookNotifier();
        this.uploadExecutor = Executors.newSingleThreadExecutor(new DaemonThreadFactory("Herbert-Upload"));
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
            String usernameHash = HashUtil.sha256Hex(player.getName());
            SessionHeader header = new SessionHeader(HerbertConstants.SCHEMA_VERSION, HerbertConstants.MOD_VERSION,
                    newSessionId, startTimestamp, usernameHash);

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
        writer = null;
        currentLogFile = null;

        chat("Session stopped. (uploading...)");
        FMLLog.log(HerbertConstants.MOD_NAME, Level.INFO, "Session %s stopped (%s), %d ticks over %ds",
                sessionId, reason, finalTickCount, durationSeconds);

        uploadExecutor.submit(new Runnable() {
            @Override
            public void run() {
                try {
                    writerToClose.close();
                } catch (IOException e) {
                    FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Failed to close session log file");
                }
                performUploadFlow(fileToUpload, durationSeconds, finalTickCount);
            }
        });
    }

    /**
     * Executes the pastes.dev + Discord upload flow for a completed session. Always runs on the
     * upload executor's background thread, never on the client thread. On any failure, the local
     * JSONL file is left intact on disk; the failure is logged and reported to the player via
     * chat.
     */
    private void performUploadFlow(File file, long durationSeconds, long tickCount) {
        if (config.isDryRunMode()) {
            scheduleChat("Herbert: dry-run mode enabled, upload skipped. Log saved at " + file.getAbsolutePath());
            return;
        }
        String webhookUrl = config.getDiscordWebhookUrl();
        if (webhookUrl == null || webhookUrl.isEmpty()) {
            scheduleChat("Herbert: no webhook configured, skipping upload");
            return;
        }

        UploadResult result = uploadToPastesDev(file);
        if (!result.isSuccess()) {
            scheduleChat("Upload failed: " + result.getErrorMessage() + " (log kept at " + file.getAbsolutePath() + ")");
            return;
        }

        try {
            discordWebhookNotifier.notify(webhookUrl, result.getPasteUrl(), durationSeconds, tickCount,
                    HerbertConstants.SCHEMA_VERSION, HerbertConstants.MOD_VERSION);
        } catch (Exception discordError) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, discordError,
                    "Discord webhook notification failed after a successful pastes.dev upload (paste: %s)", result.getPasteUrl());
        }

        scheduleChat("Upload complete: " + result.getPasteUrl());
    }

    /**
     * Reads the completed session file and uploads it to pastes.dev, wrapping any failure
     * (file I/O, network error, non-2xx response, malformed response) into a failed
     * {@link UploadResult} rather than throwing, so the caller can uniformly report success or
     * failure to the player. The local file is never touched/deleted here regardless of outcome.
     */
    private UploadResult uploadToPastesDev(File file) {
        try {
            String content = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
            String pasteUrl = pastesDevUploader.upload(content);
            return UploadResult.success(pasteUrl);
        } catch (Exception e) {
            FMLLog.log(HerbertConstants.MOD_NAME, Level.ERROR, e, "Session upload failed for %s", file.getAbsolutePath());
            return UploadResult.failure(String.valueOf(e.getMessage()));
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
