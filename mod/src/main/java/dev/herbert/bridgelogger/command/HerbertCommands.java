package dev.herbert.bridgelogger.command;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import dev.herbert.bridgelogger.session.SessionManager;
import dev.herbert.bridgelogger.util.HerbertConstants;
import net.minecraft.command.CommandBase;
import net.minecraft.command.ICommandSender;
import net.minecraft.util.ChatComponentText;

/**
 * Registers and handles the {@code /herbert start|stop|status} client-side chat command.
 *
 * <p>This is the reliable manual fallback for controlling recording sessions, since the
 * automatic Bridge-duel detection heuristics in {@link dev.herbert.bridgelogger.capture.BridgeDetector}
 * may need per-user/per-server tuning and can occasionally mis-detect or fail to detect a match.</p>
 */
public final class HerbertCommands extends CommandBase {

    private static final String USAGE = "/herbert <start|stop|status>";
    private static final List<String> SUBCOMMANDS = Arrays.asList("start", "stop", "status");

    private final SessionManager sessionManager;

    /**
     * Creates the command handler bound to the given session manager.
     *
     * @param sessionManager the session manager whose lifecycle this command controls; must not be {@code null}
     */
    public HerbertCommands(SessionManager sessionManager) {
        if (sessionManager == null) {
            throw new IllegalArgumentException("sessionManager must not be null");
        }
        this.sessionManager = sessionManager;
    }

    /** {@inheritDoc} */
    @Override
    public String getCommandName() {
        return HerbertConstants.COMMAND_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public String getCommandUsage(ICommandSender sender) {
        return USAGE;
    }

    /**
     * Client-side chat commands have no meaningful permission concept (there is no server to
     * enforce it against); always allow the local player to use this command.
     *
     * @param sender the command sender
     * @return always {@code true}
     */
    @Override
    public boolean canCommandSenderUseCommand(ICommandSender sender) {
        return true;
    }

    /**
     * Dispatches {@code start}/{@code stop}/{@code status} to the {@link SessionManager}. Any
     * unrecognized or missing subcommand prints the usage string rather than throwing, since an
     * uncaught exception here would surface as an ugly client-side error to the player.
     *
     * @param sender the command sender (the local player, for client commands)
     * @param args the command arguments, expected to be exactly one of start/stop/status
     */
    @Override
    public void processCommand(ICommandSender sender, String[] args) {
        if (args.length != 1) {
            sender.addChatMessage(new ChatComponentText(USAGE));
            return;
        }
        String sub = args[0].toLowerCase(Locale.ROOT);
        if (!SUBCOMMANDS.contains(sub)) {
            sender.addChatMessage(new ChatComponentText(USAGE));
            return;
        }
        try {
            if ("start".equals(sub)) {
                sessionManager.startManual();
            } else if ("stop".equals(sub)) {
                sessionManager.stopManual();
            } else {
                sender.addChatMessage(new ChatComponentText(sessionManager.getStatusText()));
            }
        } catch (Exception e) {
            sender.addChatMessage(new ChatComponentText("Herbert: command failed (" + e.getMessage() + ")"));
        }
    }
}
