package dev.herbert.bridgelogger.upload;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import dev.herbert.bridgelogger.util.HerbertConstants;

/**
 * Minimal HTTP client that posts a standard Discord webhook message announcing a completed
 * BridgeLogger session upload.
 *
 * <p>Uses only {@link HttpURLConnection}, already part of the JRE. Performs blocking network
 * I/O and must therefore only ever be invoked from a background thread/executor.</p>
 */
public final class DiscordWebhookNotifier {

    /**
     * Posts a message to the given Discord webhook announcing a completed session upload.
     *
     * @param webhookUrl the configured Discord webhook URL; must not be {@code null} or empty
     * @param pasteUrl the full {@code https://pastes.dev/{key}} URL of the uploaded session log; must not be {@code null}
     * @param durationSeconds wall-clock duration of the recorded session, in seconds
     * @param tickCount number of ticks recorded in the session
     * @param schemaVersion the JSONL schema version the session was written with
     * @param modVersion the BridgeLogger mod version that recorded the session
     * @throws IOException if the network request fails or Discord responds with a non-2xx status
     */
    public void notify(String webhookUrl, String pasteUrl, long durationSeconds, long tickCount, String schemaVersion,
            String modVersion) throws IOException {
        if (webhookUrl == null || webhookUrl.isEmpty()) {
            throw new IllegalArgumentException("webhookUrl must not be null or empty");
        }
        if (pasteUrl == null) {
            throw new IllegalArgumentException("pasteUrl must not be null");
        }

        String label = String.format("Herbert session upload — %ds / %d ticks", durationSeconds, tickCount);
        String payload = buildPayload(label, pasteUrl, durationSeconds, tickCount, schemaVersion, modVersion);

        HttpURLConnection connection = null;
        try {
            URL url = new URL(webhookUrl);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setConnectTimeout(HerbertConstants.HTTP_CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(HerbertConstants.HTTP_READ_TIMEOUT_MS);
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            connection.setRequestProperty("User-Agent", HerbertConstants.MOD_NAME + "/" + HerbertConstants.MOD_VERSION);

            byte[] body = payload.getBytes(StandardCharsets.UTF_8);
            connection.setFixedLengthStreamingMode(body.length);
            try (OutputStream out = connection.getOutputStream()) {
                out.write(body);
            }

            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) {
                String errorBody = readErrorBody(connection);
                throw new IOException("Discord webhook responded with HTTP " + status + ": " + errorBody);
            }
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private String buildPayload(String label, String pasteUrl, long durationSeconds, long tickCount,
            String schemaVersion, String modVersion) {
        JsonObject embed = new JsonObject();
        embed.addProperty("title", "Herbert session upload");
        embed.addProperty("url", pasteUrl);
        embed.addProperty("color", HerbertConstants.DISCORD_EMBED_COLOR);
        embed.addProperty("description", pasteUrl);

        JsonArray fields = new JsonArray();
        fields.add(field("Duration", durationSeconds + "s", true));
        fields.add(field("Ticks", String.valueOf(tickCount), true));
        fields.add(field("Schema version", schemaVersion, true));
        fields.add(field("Mod version", modVersion, true));
        embed.add("fields", fields);

        JsonArray embeds = new JsonArray();
        embeds.add(embed);

        // The plain-text `content` field (not the embed) is what downstream consumers such as
        // the intake bot scan for a pastes.dev URL via simple text matching — Discord embed
        // fields are a separate structure that a plain-content regex scan never sees. The URL
        // therefore must appear in `content` itself, not only inside the embed below (which
        // exists purely for a nicer human-readable presentation in Discord).
        JsonObject payload = new JsonObject();
        payload.addProperty("content", label + "\n" + pasteUrl);
        payload.add("embeds", embeds);
        return payload.toString();
    }

    private JsonObject field(String name, String value, boolean inline) {
        JsonObject field = new JsonObject();
        field.addProperty("name", name);
        field.addProperty("value", value);
        field.addProperty("inline", inline);
        return field;
    }

    private String readErrorBody(HttpURLConnection connection) {
        InputStream stream = null;
        try {
            stream = connection.getErrorStream();
            if (stream == null) {
                return "";
            }
            StringBuilder builder = new StringBuilder();
            byte[] buffer = new byte[512];
            int read;
            while ((read = stream.read(buffer)) != -1) {
                builder.append(new String(buffer, 0, read, StandardCharsets.UTF_8));
            }
            return builder.toString();
        } catch (IOException e) {
            return "";
        } finally {
            if (stream != null) {
                try {
                    stream.close();
                } catch (IOException ignored) {
                    // Nothing further can be done.
                }
            }
        }
    }
}
