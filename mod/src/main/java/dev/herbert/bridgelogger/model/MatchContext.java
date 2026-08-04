package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * Best-effort snapshot of match metadata scraped from the Hypixel scoreboard/tab list.
 *
 * <p>Hypixel's scoreboard text format is not officially documented and changes over time, so
 * every field here is nullable: {@link dev.herbert.bridgelogger.capture.MatchContextParser}
 * fills in whatever it can confidently parse and leaves the rest {@code null} rather than
 * guessing or throwing.</p>
 */
public final class MatchContext {

    /** Local player's own score/goals, or {@code null} if it could not be parsed. */
    @SerializedName("own_score")
    public final Integer ownScore;

    /** Opponent's score/goals, or {@code null} if it could not be parsed. */
    @SerializedName("opponent_score")
    public final Integer opponentScore;

    /** Elapsed match time in seconds, or {@code null} if it could not be parsed. */
    @SerializedName("elapsed_seconds")
    public final Integer elapsedSeconds;

    /** Best-effort kit name/id (e.g. "Iron Man"), or {@code null} if it could not be determined. */
    @SerializedName("kit")
    public final String kit;

    /**
     * Creates an immutable match context snapshot. Any parameter may be {@code null} to
     * indicate that particular field could not be extracted this tick.
     *
     * @param ownScore local player's score, or {@code null}
     * @param opponentScore opponent's score, or {@code null}
     * @param elapsedSeconds elapsed match duration in seconds, or {@code null}
     * @param kit best-effort kit name, or {@code null}
     */
    public MatchContext(Integer ownScore, Integer opponentScore, Integer elapsedSeconds, String kit) {
        this.ownScore = ownScore;
        this.opponentScore = opponentScore;
        this.elapsedSeconds = elapsedSeconds;
        this.kit = kit;
    }

    /**
     * @return {@code true} if every field is {@code null}, meaning nothing useful was parsed
     *         and this context is not worth attaching to the tick record
     */
    public boolean isEmpty() {
        return ownScore == null && opponentScore == null && elapsedSeconds == null && kit == null;
    }
}
