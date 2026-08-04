package dev.herbert.bridgelogger.serialize;

import org.apache.logging.log4j.Level;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import dev.herbert.bridgelogger.model.SessionHeader;
import dev.herbert.bridgelogger.model.TickSnapshot;
import net.minecraftforge.fml.common.FMLLog;

/**
 * Converts {@link SessionHeader} and {@link TickSnapshot} objects into single-line, compact
 * JSON strings suitable for JSON Lines (JSONL) output — one JSON object per line, no embedded
 * newlines, no pretty-printing whitespace.
 */
public final class SessionSerializer {

    private final Gson gson;

    /**
     * Creates a serializer using a compact (non-pretty-printed) Gson instance, which is
     * required for valid JSONL output (one JSON value per line).
     */
    public SessionSerializer() {
        this.gson = new GsonBuilder().disableHtmlEscaping().create();
    }

    /**
     * Serializes a session header to a single compact JSON line (without a trailing newline).
     *
     * @param header the session header to serialize; must not be {@code null}
     * @return the JSON representation of {@code header}, or {@code null} if serialization
     *         unexpectedly failed (logged via {@link FMLLog} rather than thrown, so a single bad
     *         object can never crash the caller's tick handling)
     */
    public String serializeHeader(SessionHeader header) {
        return safeToJson(header, "session header");
    }

    /**
     * Serializes a single tick snapshot to a single compact JSON line (without a trailing newline).
     *
     * @param snapshot the tick snapshot to serialize; must not be {@code null}
     * @return the JSON representation of {@code snapshot}, or {@code null} if serialization
     *         unexpectedly failed (logged via {@link FMLLog} rather than thrown)
     */
    public String serializeTick(TickSnapshot snapshot) {
        return safeToJson(snapshot, "tick snapshot");
    }

    private String safeToJson(Object value, String description) {
        if (value == null) {
            return null;
        }
        try {
            String json = gson.toJson(value);
            // Defensive: JSONL requires exactly one record per line. Gson's compact writer never
            // emits raw newlines for our POJOs, but strip any just in case a future field (e.g. a
            // free-text scoreboard string) somehow smuggled one through.
            return json.replace("\n", "").replace("\r", "");
        } catch (Exception e) {
            FMLLog.log("BridgeLogger", Level.ERROR, e, "Failed to serialize %s to JSON", description);
            return null;
        }
    }
}
