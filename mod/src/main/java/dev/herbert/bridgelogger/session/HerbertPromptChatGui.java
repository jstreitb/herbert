package dev.herbert.bridgelogger.session;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiChat;

/**
 * A drop-in replacement for vanilla's chat GUI ({@link GuiChat}), swapped in for exactly one
 * chat interaction via {@link SessionManager}'s {@code GuiOpenEvent} handler while a session-end
 * "display your username publicly?" prompt is awaiting a response.
 *
 * <p><b>This is a UX/privacy mechanism, not gameplay automation:</b> it never injects, replays,
 * or synthesizes any input (consistent with the transparency statement in
 * {@link dev.herbert.bridgelogger.BridgeLoggerMod}). It only intercepts the message the player
 * <i>themselves</i> chose to type into the chat box they opened, in the narrow window right after
 * a Herbert prompt was printed, so their raw "y"/"n" answer never leaves the client as a public
 * chat packet -- see {@link #sendChatMessage(String)}.</p>
 *
 * <p>Forge 1.8.9 has no dedicated "outgoing chat" event to cancel (only
 * {@code net.minecraftforge.event.ServerChatEvent}, which is server-side and irrelevant to a
 * client mod talking to a remote server). Overriding {@link GuiChat}'s inherited
 * {@code sendChatMessage(String)} -- the exact method vanilla's {@code GuiChat.keyTyped} calls on
 * Enter, before any packet is constructed or sent -- is this build's equivalent interception
 * point.</p>
 */
final class HerbertPromptChatGui extends GuiChat {

    private final SessionManager owner;

    /**
     * Creates a chat GUI that hands its next submitted message to {@code owner} instead of
     * sending it to the server.
     *
     * @param owner the session manager to notify once the player submits a message; must not be
     *        {@code null}
     */
    HerbertPromptChatGui(SessionManager owner) {
        super();
        if (owner == null) {
            throw new IllegalArgumentException("owner must not be null");
        }
        this.owner = owner;
    }

    /**
     * Called by vanilla's {@code GuiChat.keyTyped} when the player presses Enter with non-empty
     * text in the chat box. Deliberately does <b>not</b> call {@code super.sendChatMessage(...)}
     * (which would forward the text to {@code EntityPlayerSP.sendChatMessage} and from there to
     * the server as a real chat packet) -- instead it closes the chat box itself and routes the
     * raw text to {@link SessionManager#onUsernamePromptChatCaptured(String)} for validation.
     *
     * @param message the trimmed, non-empty text the player typed
     */
    @Override
    public void sendChatMessage(String message) {
        Minecraft.getMinecraft().displayGuiScreen(null);
        owner.onUsernamePromptChatCaptured(message);
    }
}
