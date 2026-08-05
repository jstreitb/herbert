// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger;

import java.io.File;

import dev.herbert.bridgelogger.command.HerbertCommands;
import dev.herbert.bridgelogger.config.HerbertConfig;
import dev.herbert.bridgelogger.session.SessionManager;
import dev.herbert.bridgelogger.util.HerbertConstants;
import net.minecraftforge.client.ClientCommandHandler;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.FMLCommonHandler;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;

/**
 * Main mod entry point for BridgeLogger, part of the Herbert project.
 *
 * <p>BridgeLogger is a purely observational data-collection mod: it passively records
 * synchronized player-state, block-environment, opponent-state, and input/action data during
 * Hypixel Bridge duels at a fixed tick resolution, then uploads the resulting JSONL log for use
 * in training an imitation-learning model.</p>
 *
 * <p><b>Transparency statement:</b> this mod never sends game packets, never simulates input,
 * never automates any game action, and never injects input back into the game client. It only
 * reads client-side state that Minecraft already exposes locally (player position, nearby
 * blocks, already-synced entity data, and the local player's own input state) purely to record
 * it for later offline training. There is no autoclicker, no aim assist, no scaffold/bridge
 * assist, and no other gameplay automation anywhere in this codebase.</p>
 *
 * <p>This class only wires together the mod's lifecycle: it loads the config in the
 * pre-initialization phase, then constructs and registers the {@link SessionManager} (which
 * owns all actual recording logic) and the {@link HerbertCommands} client command during the
 * initialization phase. This mod is client-only and registers no server-side behavior.</p>
 */
@Mod(modid = HerbertConstants.MOD_ID, name = HerbertConstants.MOD_NAME, version = HerbertConstants.MOD_VERSION)
public final class BridgeLoggerMod {

    private final HerbertConfig config = new HerbertConfig();
    private SessionManager sessionManager;

    /**
     * Forge pre-initialization phase: loads (or creates, with defaults) the mod configuration
     * file at the path Forge suggests for this mod id.
     *
     * @param event the FML pre-initialization event, used only for its suggested config file path
     */
    @EventHandler
    public void preInit(FMLPreInitializationEvent event) {
        File configFile = event.getSuggestedConfigurationFile();
        config.load(configFile);
    }

    /**
     * Forge initialization phase: constructs the {@link SessionManager}, registers it on both
     * Forge event buses it needs (the main game event bus for tick/interaction events, and the
     * FML event bus for connect/disconnect network events), and registers the {@code /herbert}
     * client command.
     *
     * @param event the FML initialization event
     */
    @EventHandler
    public void init(FMLInitializationEvent event) {
        sessionManager = new SessionManager(config);
        MinecraftForge.EVENT_BUS.register(sessionManager);
        FMLCommonHandler.instance().bus().register(sessionManager);
        ClientCommandHandler.instance.registerCommand(new HerbertCommands(sessionManager));
    }
}
