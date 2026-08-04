package dev.herbert.bridgelogger.upload;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

import dev.herbert.bridgelogger.util.HerbertConstants;

/**
 * Minimal HTTP client for uploading a completed session log to
 * <a href="https://pastes.dev">pastes.dev</a> as an anonymous paste.
 *
 * <p>Uses only {@link HttpURLConnection}, already part of the JRE, to avoid pulling in any new
 * dependency. Every method here performs blocking network I/O and must therefore only ever be
 * invoked from a background thread/executor — never from the client thread.</p>
 */
public final class PastesDevUploader {

    /**
     * Uploads {@code content} to pastes.dev and returns the resulting viewable paste URL.
     *
     * @param content the full text content to upload (the session's JSONL file contents); must not be {@code null}
     * @return the resulting {@code https://pastes.dev/{key}} URL
     * @throws IOException if the network request fails, the server responds with a non-2xx
     *         status, or the response body cannot be parsed as JSON containing a paste key
     */
    public String upload(String content) throws IOException {
        if (content == null) {
            throw new IllegalArgumentException("content must not be null");
        }

        HttpURLConnection connection = null;
        try {
            URL url = new URL(HerbertConstants.PASTES_DEV_POST_URL);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setConnectTimeout(HerbertConstants.HTTP_CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(HerbertConstants.HTTP_READ_TIMEOUT_MS);
            connection.setRequestProperty("Content-Type", HerbertConstants.PASTES_DEV_CONTENT_TYPE + "; charset=utf-8");
            connection.setRequestProperty("User-Agent", HerbertConstants.MOD_NAME + "/" + HerbertConstants.MOD_VERSION);

            byte[] body = content.getBytes(StandardCharsets.UTF_8);
            connection.setFixedLengthStreamingMode(body.length);
            try (OutputStream out = connection.getOutputStream()) {
                out.write(body);
            }

            int status = connection.getResponseCode();
            String responseBody = readBody(connection, status);

            if (status < 200 || status >= 300) {
                throw new IOException("pastes.dev responded with HTTP " + status + ": " + truncate(responseBody, 500));
            }

            String key = extractKey(responseBody);
            if (key == null || key.isEmpty()) {
                throw new IOException("pastes.dev response did not contain a paste key: " + truncate(responseBody, 500));
            }
            return HerbertConstants.PASTES_DEV_VIEW_URL_PREFIX + key;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private String extractKey(String responseBody) {
        try {
            JsonElement parsed = new JsonParser().parse(responseBody);
            if (!parsed.isJsonObject()) {
                return null;
            }
            JsonObject object = parsed.getAsJsonObject();
            if (object.has("key") && object.get("key").isJsonPrimitive()) {
                return object.get("key").getAsString();
            }
            return null;
        } catch (JsonSyntaxException | IllegalStateException e) {
            return null;
        }
    }

    private String readBody(HttpURLConnection connection, int status) {
        InputStream stream = null;
        try {
            stream = (status >= 200 && status < 300) ? connection.getInputStream() : connection.getErrorStream();
            if (stream == null) {
                return "";
            }
            StringBuilder builder = new StringBuilder();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
            char[] buffer = new char[1024];
            int read;
            while ((read = reader.read(buffer)) != -1) {
                builder.append(buffer, 0, read);
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

    private String truncate(String value, int maxLength) {
        if (value == null) {
            return "";
        }
        return value.length() <= maxLength ? value : value.substring(0, maxLength) + "...";
    }
}
