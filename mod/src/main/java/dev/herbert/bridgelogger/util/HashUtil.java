// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.util;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Small cryptographic hashing helper used to anonymize player identity before it is ever
 * written to disk or uploaded anywhere.
 *
 * <p><b>Privacy contract:</b> BridgeLogger must never persist a raw Minecraft username. Every
 * session header instead stores {@link #sha256Hex(String)} of the username. This class is the
 * single place that implements that hashing so the rule is easy to audit.</p>
 */
public final class HashUtil {

    /** Digest algorithm used to anonymize usernames. */
    private static final String ALGORITHM = "SHA-256";

    private HashUtil() {
        // Static utility class; never instantiated.
    }

    /**
     * Computes the lowercase hexadecimal SHA-256 digest of the given input string.
     *
     * @param input the plaintext to hash (e.g. a player's username); must not be {@code null}
     * @return a 64-character lowercase hex string representing the SHA-256 digest of {@code input}
     * @throws IllegalArgumentException if {@code input} is {@code null}
     * @throws IllegalStateException if the JVM does not provide a SHA-256 {@link MessageDigest}
     *         implementation, which would indicate a broken Java installation since SHA-256 is
     *         mandated by the Java platform specification
     */
    public static String sha256Hex(String input) {
        if (input == null) {
            throw new IllegalArgumentException("input must not be null");
        }
        try {
            MessageDigest digest = MessageDigest.getInstance(ALGORITHM);
            byte[] hashBytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            return toHex(hashBytes);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 algorithm not available on this JVM", e);
        }
    }

    /**
     * Converts a byte array into a lowercase hexadecimal string representation.
     *
     * @param bytes the bytes to encode; must not be {@code null}
     * @return a lowercase hex string twice the length of {@code bytes}
     */
    private static String toHex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) {
                builder.append('0');
            }
            builder.append(hex);
        }
        return builder.toString();
    }
}
