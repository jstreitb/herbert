package dev.herbert.bridgelogger.upload;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Locale;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import dev.herbert.bridgelogger.serialize.SessionChunk;
import dev.herbert.bridgelogger.serialize.SessionChunker;
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

    /**
     * Uploads one split session chunk (see {@link dev.herbert.bridgelogger.serialize.SessionChunker})
     * directly to the given Discord webhook as a file attachment, with its own embed describing
     * which part this is, the tick range it covers, its file size, and the usual session metadata.
     *
     * <p>Unlike {@link #notify}, which posts a plain JSON message linking to an already-uploaded
     * pastes.dev paste, this performs a {@code multipart/form-data} request carrying the chunk
     * file's bytes directly -- chunked sessions are never uploaded to pastes.dev at all (see
     * {@code mod/README.md}'s "Chunked uploads" section for why, and for the resulting known gap
     * with {@code /bot}'s current pastes.dev-URL-scanning intake validator).</p>
     *
     * @param webhookUrl the configured Discord webhook URL; must not be {@code null} or empty
     * @param chunk the chunk to upload; must not be {@code null}
     * @param sessionId the session's UUID (shown in the embed; identical across every chunk)
     * @param durationSeconds wall-clock duration of the whole recorded session, in seconds
     * @param tickCount total number of ticks recorded across the whole session (not just this chunk)
     * @param schemaVersion the JSONL schema version the session was written with
     * @param modVersion the BridgeLogger mod version that recorded the session
     * @throws IOException if the chunk file cannot be read, the network request fails, or
     *         Discord responds with a non-2xx status
     */
    public void uploadChunk(String webhookUrl, SessionChunk chunk, String sessionId, long durationSeconds,
            long tickCount, String schemaVersion, String modVersion) throws IOException {
        if (webhookUrl == null || webhookUrl.isEmpty()) {
            throw new IllegalArgumentException("webhookUrl must not be null or empty");
        }
        if (chunk == null) {
            throw new IllegalArgumentException("chunk must not be null");
        }

        byte[] fileBytes = Files.readAllBytes(chunk.getFile().toPath());
        String payloadJson = buildChunkPayload(chunk, sessionId, durationSeconds, tickCount, schemaVersion, modVersion);
        String boundary = "----HerbertChunk" + System.nanoTime();
        byte[] body = buildMultipartBody(boundary, payloadJson, chunk.getFile().getName(), fileBytes);

        HttpURLConnection connection = null;
        try {
            URL url = new URL(webhookUrl);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setConnectTimeout(HerbertConstants.HTTP_CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(HerbertConstants.HTTP_READ_TIMEOUT_MS);
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
            connection.setRequestProperty("User-Agent", HerbertConstants.MOD_NAME + "/" + HerbertConstants.MOD_VERSION);

            connection.setFixedLengthStreamingMode(body.length);
            try (OutputStream out = connection.getOutputStream()) {
                out.write(body);
            }

            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) {
                String errorBody = readErrorBody(connection);
                throw new IOException("Discord webhook file upload responded with HTTP " + status + ": " + errorBody);
            }
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private String buildChunkPayload(SessionChunk chunk, String sessionId, long durationSeconds, long tickCount,
            String schemaVersion, String modVersion) {
        int partNumber = chunk.getPartNumber();
        int chunkTotal = chunk.getChunkTotal();

        JsonObject embed = new JsonObject();
        embed.addProperty("title", "Herbert session — part " + partNumber + " of " + chunkTotal);
        embed.addProperty("color", HerbertConstants.DISCORD_EMBED_COLOR);

        JsonArray fields = new JsonArray();
        fields.add(field("Session ID", sessionId, false));
        fields.add(field("Tick range (this part)", tickRangeText(chunk), true));
        fields.add(field("Chunk size", formatBytes(chunk.getSizeBytes()), true));
        fields.add(field("Duration", durationSeconds + "s", true));
        fields.add(field("Ticks (session total)", String.valueOf(tickCount), true));
        fields.add(field("Schema version", schemaVersion, true));
        fields.add(field("Mod version", modVersion, true));
        embed.add("fields", fields);

        JsonArray embeds = new JsonArray();
        embeds.add(embed);

        JsonObject payload = new JsonObject();
        payload.addProperty("content",
                String.format(Locale.ROOT, "Herbert session upload — part %d of %d (session %s)", partNumber, chunkTotal, sessionId));
        payload.add("embeds", embeds);
        return payload.toString();
    }

    private String tickRangeText(SessionChunk chunk) {
        if (chunk.getFirstTick() == SessionChunker.NO_TICKS) {
            return "(no ticks)";
        }
        return chunk.getFirstTick() + " .. " + chunk.getLastTick();
    }

    private String formatBytes(long bytes) {
        return String.format(Locale.ROOT, "%.2f MB", bytes / (1024.0 * 1024.0));
    }

    /**
     * Builds a {@code multipart/form-data} body with two parts: a {@code payload_json} field
     * (Discord's convention for the message JSON when a file is attached) and a {@code files[0]}
     * field carrying the raw chunk file bytes.
     */
    private byte[] buildMultipartBody(String boundary, String payloadJson, String fileName, byte[] fileBytes) throws IOException {
        String lineBreak = "\r\n";
        ByteArrayOutputStream buffer = new ByteArrayOutputStream(fileBytes.length + 1024);

        buffer.write(("--" + boundary + lineBreak).getBytes(StandardCharsets.UTF_8));
        buffer.write(("Content-Disposition: form-data; name=\"payload_json\"" + lineBreak).getBytes(StandardCharsets.UTF_8));
        buffer.write(("Content-Type: application/json" + lineBreak + lineBreak).getBytes(StandardCharsets.UTF_8));
        buffer.write(payloadJson.getBytes(StandardCharsets.UTF_8));
        buffer.write(lineBreak.getBytes(StandardCharsets.UTF_8));

        buffer.write(("--" + boundary + lineBreak).getBytes(StandardCharsets.UTF_8));
        buffer.write(("Content-Disposition: form-data; name=\"files[0]\"; filename=\"" + fileName + "\"" + lineBreak)
                .getBytes(StandardCharsets.UTF_8));
        buffer.write(("Content-Type: application/octet-stream" + lineBreak + lineBreak).getBytes(StandardCharsets.UTF_8));
        buffer.write(fileBytes);
        buffer.write(lineBreak.getBytes(StandardCharsets.UTF_8));

        buffer.write(("--" + boundary + "--" + lineBreak).getBytes(StandardCharsets.UTF_8));
        return buffer.toByteArray();
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
