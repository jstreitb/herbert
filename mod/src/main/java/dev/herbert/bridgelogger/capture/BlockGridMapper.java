package dev.herbert.bridgelogger.capture;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import dev.herbert.bridgelogger.model.BlockCategory;
import dev.herbert.bridgelogger.util.HerbertConstants;
import net.minecraft.block.Block;
import net.minecraft.block.material.Material;
import net.minecraft.init.Blocks;
import net.minecraft.world.World;

/**
 * Maps a single world block to its coarse {@link BlockCategory}.
 *
 * <p>This class intentionally exposes pure, static classification logic with no dependency on
 * a running client (no {@code Minecraft.getMinecraft()} calls) so it is trivially unit
 * testable: given a {@link Block} and a Y coordinate, {@link #mapBlockToCategory} always
 * returns the same result. The {@link World}/coordinates are accepted for API stability and
 * future extensibility (e.g. neighbor-aware classification) but the current implementation only
 * needs the block itself and the Y coordinate.</p>
 *
 * <p>No raw numeric block IDs are used anywhere in this class; every block referenced below is
 * a named constant from Mojang/Forge's {@link Blocks} registry class, grouped into a single
 * documented lookup table.</p>
 */
public final class BlockGridMapper {

    /**
     * Full-cube, "bridge material" style blocks: things a player would plausibly stand on or
     * place while bridging on a Hypixel Bridge map (wool, terracotta/clay, common building
     * blocks). This set is the single place to extend if the community identifies more blocks
     * that should count as {@link BlockCategory#SOLID_BRIDGEABLE}.
     */
    private static final Set<Block> BRIDGEABLE_BLOCKS = buildBridgeableBlockSet();

    private BlockGridMapper() {
        // Static utility class; never instantiated.
    }

    private static Set<Block> buildBridgeableBlockSet() {
        Set<Block> blocks = new HashSet<Block>();
        blocks.add(Blocks.wool);
        blocks.add(Blocks.clay);
        blocks.add(Blocks.hardened_clay);
        blocks.add(Blocks.stained_hardened_clay);
        blocks.add(Blocks.planks);
        blocks.add(Blocks.stone);
        blocks.add(Blocks.cobblestone);
        blocks.add(Blocks.mossy_cobblestone);
        blocks.add(Blocks.brick_block);
        blocks.add(Blocks.nether_brick);
        blocks.add(Blocks.end_stone);
        blocks.add(Blocks.sandstone);
        blocks.add(Blocks.dirt);
        blocks.add(Blocks.grass);
        blocks.add(Blocks.iron_block);
        blocks.add(Blocks.gold_block);
        blocks.add(Blocks.emerald_block);
        blocks.add(Blocks.diamond_block);
        blocks.add(Blocks.coal_block);
        blocks.add(Blocks.quartz_block);
        blocks.add(Blocks.snow);
        blocks.add(Blocks.ice);
        blocks.add(Blocks.packed_ice);
        return Collections.unmodifiableSet(blocks);
    }

    /**
     * Classifies a world block into a coarse {@link BlockCategory}, using the default void-Y
     * threshold ({@link HerbertConstants#DEFAULT_VOID_THRESHOLD_Y}).
     *
     * @param block the block to classify, as returned by {@code World.getBlockState(pos).getBlock()}; may be {@code null}
     * @param x world X coordinate of the block (currently unused, reserved for future neighbor-aware logic)
     * @param y world Y coordinate of the block; used to distinguish {@link BlockCategory#AIR} from {@link BlockCategory#VOID}
     * @param z world Z coordinate of the block (currently unused, reserved for future neighbor-aware logic)
     * @param world the world the block belongs to (currently unused, reserved for future neighbor-aware logic); may be {@code null}
     * @return the block's coarse category, never {@code null}
     */
    public static BlockCategory mapBlockToCategory(Block block, int x, int y, int z, World world) {
        return mapBlockToCategory(block, x, y, z, world, HerbertConstants.DEFAULT_VOID_THRESHOLD_Y);
    }

    /**
     * Classifies a world block into a coarse {@link BlockCategory}, using an explicit void-Y
     * threshold. This is the pure, fully deterministic overload intended for unit testing.
     *
     * @param block the block to classify; may be {@code null} (treated the same as air)
     * @param x world X coordinate of the block (currently unused, reserved for future neighbor-aware logic)
     * @param y world Y coordinate of the block; compared against {@code voidThresholdY}
     * @param z world Z coordinate of the block (currently unused, reserved for future neighbor-aware logic)
     * @param world the world the block belongs to (currently unused, reserved for future neighbor-aware logic); may be {@code null}
     * @param voidThresholdY world Y coordinate at/below which air is classified as {@link BlockCategory#VOID} instead of {@link BlockCategory#AIR}
     * @return the block's coarse category, never {@code null}
     */
    public static BlockCategory mapBlockToCategory(Block block, int x, int y, int z, World world, int voidThresholdY) {
        if (block == null || block == Blocks.air) {
            return y <= voidThresholdY ? BlockCategory.VOID : BlockCategory.AIR;
        }

        Material material = block.getMaterial();
        if (material == Material.water || material == Material.lava) {
            return BlockCategory.LIQUID;
        }

        if (BRIDGEABLE_BLOCKS.contains(block)) {
            return BlockCategory.SOLID_BRIDGEABLE;
        }

        return BlockCategory.OTHER_SOLID;
    }
}
