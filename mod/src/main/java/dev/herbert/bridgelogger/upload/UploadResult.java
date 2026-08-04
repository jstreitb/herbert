package dev.herbert.bridgelogger.upload;

/**
 * Outcome of a full upload attempt (pastes.dev POST, optionally followed by a Discord webhook
 * notification), used to drive the chat feedback shown to the player.
 */
public final class UploadResult {

    private final boolean success;
    private final String pasteUrl;
    private final String errorMessage;

    private UploadResult(boolean success, String pasteUrl, String errorMessage) {
        this.success = success;
        this.pasteUrl = pasteUrl;
        this.errorMessage = errorMessage;
    }

    /**
     * @param pasteUrl the resulting {@code https://pastes.dev/{key}} URL; must not be {@code null}
     * @return a successful result carrying the paste URL
     */
    public static UploadResult success(String pasteUrl) {
        if (pasteUrl == null) {
            throw new IllegalArgumentException("pasteUrl must not be null for a successful result");
        }
        return new UploadResult(true, pasteUrl, null);
    }

    /**
     * @param errorMessage human-readable failure reason; must not be {@code null}
     * @return a failed result carrying the error reason
     */
    public static UploadResult failure(String errorMessage) {
        if (errorMessage == null) {
            throw new IllegalArgumentException("errorMessage must not be null for a failure result");
        }
        return new UploadResult(false, null, errorMessage);
    }

    /** @return {@code true} if the paste was uploaded successfully */
    public boolean isSuccess() {
        return success;
    }

    /** @return the resulting pastes.dev URL, or {@code null} if {@link #isSuccess()} is {@code false} */
    public String getPasteUrl() {
        return pasteUrl;
    }

    /** @return a human-readable failure reason, or {@code null} if {@link #isSuccess()} is {@code true} */
    public String getErrorMessage() {
        return errorMessage;
    }
}
