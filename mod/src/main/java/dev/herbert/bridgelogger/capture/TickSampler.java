package dev.herbert.bridgelogger.capture;

import java.time.Instant;
import java.util.List;

import dev.herbert.bridgelogger.config.HerbertConfig;
import dev.herbert.bridgelogger.model.BlockCategory;
import dev.herbert.bridgelogger.model.BlockGridSnapshot;
import dev.herbert.bridgelogger.model.HeldItemState;
import dev.herbert.bridgelogger.model.InputSnapshot;
import dev.herbert.bridgelogger.model.MatchContext;
import dev.herbert.bridgelogger.model.OpponentSnapshot;
import dev.herbert.bridgelogger.model.PlayerState;
import dev.herbert.bridgelogger.model.TickSnapshot;
import dev.herbert.bridgelogger.util.HerbertConstants;
import dev.herbert.bridgelogger.util.ScoreboardUtil;
import net.minecraft.block.Block;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.scoreboard.Scoreboard;
import net.minecraft.scoreboard.ScorePlayerTeam;
import net.minecraft.util.BlockPos;
import net.minecraft.util.MathHelper;
import net.minecraft.world.World;

/**
 * Collects a single {@link TickSnapshot} of everything BridgeLogger observes about the local
 * player, the world around them, the nearest opponent, and match context.
 *
 * <p>Instances are stateful only with respect to (a) rotation from the previous sampled tick,
 * used to compute {@code deltaYaw}/{@code deltaPitch}, and (b) pending attack/place input
 * events reported by Forge event handlers between samples. Everything else is read fresh from
 * the live game state each call to {@link #sample(long, EntityPlayerSP, World)}.</p>
 *
 * <p>Every optional sub-section (block grid, opponent lookup, match context, held item) is
 * individually guarded so that a failure in one does not prevent the rest of the tick from
 * being recorded, in line with the project's "never let logging disrupt or crash the game"
 * requirement.</p>
 */
public final class TickSampler {

    private final HerbertConfig config;

    private boolean havePreviousRotation = false;
    private float previousYaw;
    private float previousPitch;

    private volatile boolean pendingAttack = false;
    private volatile String pendingAttackTargetType = null;

    private volatile boolean pendingPlace = false;
    private volatile String pendingPlaceBlockType = null;
    private volatile Integer pendingPlaceX = null;
    private volatile Integer pendingPlaceY = null;
    private volatile Integer pendingPlaceZ = null;

    /**
     * Creates a new sampler bound to the given (live-updating) config wrapper.
     *
     * @param config the mod configuration, used for block grid dimensions, void threshold, and
     *        opponent/scoreboard toggles; must not be {@code null}
     */
    public TickSampler(HerbertConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        this.config = config;
    }

    /**
     * Resets all inter-tick state (previous rotation, pending input events). Should be called
     * whenever a new recording session starts so the first sampled tick doesn't report a bogus
     * rotation delta or carry over stale input events from before recording began.
     */
    public void reset() {
        havePreviousRotation = false;
        clearPendingInput();
    }

    /**
     * Records that a left-click/attack event occurred since the last sample. Intended to be
     * called from a Forge {@code AttackEntityEvent} handler. Thread-safe with respect to
     * {@link #sample}, which always runs on the client thread alongside tick events, but this
     * method itself may be invoked from the same thread synchronously with event dispatch.
     *
     * @param targetType a coarse description of what was hit (e.g. the target entity's simple
     *        class name), or {@code null} if unknown
     */
    public void recordAttack(String targetType) {
        pendingAttack = true;
        pendingAttackTargetType = targetType;
    }

    /**
     * Records that a right-click block-place event occurred since the last sample. Intended to
     * be called from a Forge {@code PlayerInteractEvent} handler.
     *
     * @param blockType registry name of the placed block, or {@code null} if unresolvable
     * @param pos the world position the block was placed at, or {@code null} if unresolvable
     */
    public void recordPlace(String blockType, BlockPos pos) {
        pendingPlace = true;
        pendingPlaceBlockType = blockType;
        if (pos != null) {
            pendingPlaceX = pos.getX();
            pendingPlaceY = pos.getY();
            pendingPlaceZ = pos.getZ();
        } else {
            pendingPlaceX = null;
            pendingPlaceY = null;
            pendingPlaceZ = null;
        }
    }

    /**
     * Builds a complete {@link TickSnapshot} for the current game state.
     *
     * @param tickIndex the session-relative sample index to embed in the snapshot
     * @param player the local player; must not be {@code null}
     * @param world the local player's current world; must not be {@code null}
     * @return a fully populated snapshot; individual optional sections fall back to {@code null}
     *         (or, for required-non-null sections, a safe default) if they cannot be computed
     * @throws IllegalArgumentException if {@code player} or {@code world} is {@code null}
     */
    public TickSnapshot sample(long tickIndex, EntityPlayerSP player, World world) {
        if (player == null || world == null) {
            throw new IllegalArgumentException("player and world must not be null");
        }

        String timestamp = Instant.now().toString();
        PlayerState playerState = buildPlayerState(player);
        float deltaYaw = havePreviousRotation ? wrapDegrees(player.rotationYaw - previousYaw) : 0f;
        float deltaPitch = havePreviousRotation ? player.rotationPitch - previousPitch : 0f;
        previousYaw = player.rotationYaw;
        previousPitch = player.rotationPitch;
        havePreviousRotation = true;

        BlockGridSnapshot blockGrid = safeBuildBlockGrid(player, world);
        HeldItemState heldItem = safeBuildHeldItem(player);
        OpponentSnapshot opponent = config.isOpponentTrackingEnabled() ? safeFindOpponent(player, world) : null;
        MatchContext matchContext = config.isScoreboardParsingEnabled() ? safeBuildMatchContext(player, world) : null;
        InputSnapshot input = buildInputSnapshot(player, deltaYaw, deltaPitch);

        return new TickSnapshot(tickIndex, timestamp, playerState, blockGrid, heldItem, opponent, matchContext, input);
    }

    private PlayerState buildPlayerState(EntityPlayerSP player) {
        return new PlayerState(
                player.posX, player.posY, player.posZ,
                player.motionX, player.motionY, player.motionZ,
                player.rotationYaw, player.rotationPitch,
                player.onGround, player.isSneaking(),
                player.getHealth(), player.getFoodStats().getFoodLevel());
    }

    private BlockGridSnapshot safeBuildBlockGrid(EntityPlayerSP player, World world) {
        int width = config.getBlockGridWidth();
        int height = config.getBlockGridHeight();
        int depth = config.getBlockGridDepth();
        try {
            return buildBlockGrid(player, world, width, height, depth);
        } catch (Exception e) {
            BlockCategory[] fallback = new BlockCategory[width * height * depth];
            for (int i = 0; i < fallback.length; i++) {
                fallback[i] = BlockCategory.AIR;
            }
            return new BlockGridSnapshot(width, height, depth, fallback);
        }
    }

    private BlockGridSnapshot buildBlockGrid(EntityPlayerSP player, World world, int width, int height, int depth) {
        int centerX = MathHelper.floor_double(player.posX);
        int centerY = MathHelper.floor_double(player.posY);
        int centerZ = MathHelper.floor_double(player.posZ);
        int voidThresholdY = config.getVoidThresholdY();

        BlockCategory[] cells = new BlockCategory[width * height * depth];
        for (int yIndex = 0; yIndex < height; yIndex++) {
            int worldY = centerY + (yIndex - height / 2);
            for (int zIndex = 0; zIndex < depth; zIndex++) {
                int worldZ = centerZ + (zIndex - depth / 2);
                for (int xIndex = 0; xIndex < width; xIndex++) {
                    int worldX = centerX + (xIndex - width / 2);
                    BlockPos pos = new BlockPos(worldX, worldY, worldZ);
                    Block block = world.getBlockState(pos).getBlock();
                    int flatIndex = (yIndex * depth + zIndex) * width + xIndex;
                    cells[flatIndex] = BlockGridMapper.mapBlockToCategory(block, worldX, worldY, worldZ, world, voidThresholdY);
                }
            }
        }
        return new BlockGridSnapshot(width, height, depth, cells);
    }

    private HeldItemState safeBuildHeldItem(EntityPlayerSP player) {
        try {
            int slot = player.inventory.currentItem;
            ItemStack stack = player.inventory.getCurrentItem();
            if (stack == null || stack.getItem() == null) {
                return new HeldItemState(slot, null, 0);
            }
            String itemId = itemRegistryName(stack.getItem());
            return new HeldItemState(slot, itemId, stack.stackSize);
        } catch (Exception e) {
            return new HeldItemState(0, null, 0);
        }
    }

    private OpponentSnapshot safeFindOpponent(EntityPlayerSP player, World world) {
        try {
            EntityPlayer nearest = null;
            double nearestDistSq = Double.MAX_VALUE;
            double maxDistSq = HerbertConstants.OPPONENT_MAX_DISTANCE_BLOCKS * HerbertConstants.OPPONENT_MAX_DISTANCE_BLOCKS;

            List<?> players = world.playerEntities;
            for (Object candidateObj : players) {
                if (!(candidateObj instanceof EntityPlayer)) {
                    continue;
                }
                EntityPlayer candidate = (EntityPlayer) candidateObj;
                if (candidate == player) {
                    continue;
                }
                double distSq = player.getDistanceSqToEntity(candidate);
                if (distSq <= maxDistSq && distSq < nearestDistSq) {
                    nearestDistSq = distSq;
                    nearest = candidate;
                }
            }

            if (nearest == null) {
                return null;
            }

            double relX = nearest.posX - player.posX;
            double relY = nearest.posY - player.posY;
            double relZ = nearest.posZ - player.posZ;
            double relVx = nearest.motionX - player.motionX;
            double relVy = nearest.motionY - player.motionY;
            double relVz = nearest.motionZ - player.motionZ;

            ItemStack heldStack = nearest.getHeldItem();
            Item heldItem = heldStack == null ? null : heldStack.getItem();

            return new OpponentSnapshot(relX, relY, relZ, relVx, relVy, relVz, nearest.rotationYaw, nearest.rotationPitch,
                    nearest.getHealth(), HeldItemCategoryMapper.mapCategory(heldItem));
        } catch (Exception e) {
            return null;
        }
    }

    private MatchContext safeBuildMatchContext(EntityPlayerSP player, World world) {
        try {
            List<String> sidebarLines = ScoreboardUtil.getSidebarLines(world);
            String ownTeamHint = safeOwnTeamHint(player, world);
            MatchContext context = MatchContextParser.parse(sidebarLines, ownTeamHint);
            return context.isEmpty() ? null : context;
        } catch (Exception e) {
            return null;
        }
    }

    private String safeOwnTeamHint(EntityPlayerSP player, World world) {
        try {
            Scoreboard scoreboard = world.getScoreboard();
            if (scoreboard == null) {
                return null;
            }
            ScorePlayerTeam team = scoreboard.getPlayersTeam(player.getName());
            if (team == null || team.getChatFormat() == null) {
                return null;
            }
            return team.getChatFormat().name();
        } catch (Exception e) {
            return null;
        }
    }

    private InputSnapshot buildInputSnapshot(EntityPlayerSP player, float deltaYaw, float deltaPitch) {
        int forward = 0;
        int strafe = 0;
        boolean jump = false;
        boolean sneak;
        try {
            sneak = player.isSneaking();
        } catch (Exception e) {
            sneak = false;
        }
        try {
            if (player.movementInput != null) {
                forward = (int) Math.signum(player.movementInput.moveForward);
                strafe = (int) Math.signum(player.movementInput.moveStrafe);
                jump = player.movementInput.jump;
            }
        } catch (Exception e) {
            forward = 0;
            strafe = 0;
            jump = false;
        }

        boolean attackOccurred = pendingAttack;
        String attackTargetType = pendingAttackTargetType;
        boolean placeOccurred = pendingPlace;
        String placeBlockType = pendingPlaceBlockType;
        Integer placeX = pendingPlaceX;
        Integer placeY = pendingPlaceY;
        Integer placeZ = pendingPlaceZ;
        clearPendingInput();

        return new InputSnapshot(forward, strafe, jump, sneak, deltaYaw, deltaPitch, attackOccurred, attackTargetType,
                placeOccurred, placeBlockType, placeX, placeY, placeZ);
    }

    private void clearPendingInput() {
        pendingAttack = false;
        pendingAttackTargetType = null;
        pendingPlace = false;
        pendingPlaceBlockType = null;
        pendingPlaceX = null;
        pendingPlaceY = null;
        pendingPlaceZ = null;
    }

    private static String itemRegistryName(Item item) {
        try {
            Object name = Item.itemRegistry.getNameForObject(item);
            return name == null ? null : name.toString();
        } catch (Exception e) {
            return null;
        }
    }

    private static float wrapDegrees(float degrees) {
        float wrapped = degrees % 360f;
        if (wrapped >= 180f) {
            wrapped -= 360f;
        } else if (wrapped < -180f) {
            wrapped += 360f;
        }
        return wrapped;
    }
}
