// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

/** Unit tests for {@link HashUtil}. */
public class HashUtilTest {

    @Test
    public void sha256Hex_knownInput_expectsKnownDigest() {
        // Test vector verified independently via Python's hashlib.sha256("password").
        assertEquals("5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
                HashUtil.sha256Hex("password"));
    }

    @Test
    public void sha256Hex_emptyString_expectsKnownEmptyDigest() {
        // The SHA-256 digest of the empty string is a fixed, well-known constant.
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", HashUtil.sha256Hex(""));
    }

    @Test
    public void sha256Hex_isDeterministic_expectsSameDigestTwice() {
        assertEquals(HashUtil.sha256Hex("Notch"), HashUtil.sha256Hex("Notch"));
    }

    @Test
    public void sha256Hex_differentInputs_expectsDifferentDigests() {
        assertNotEquals(HashUtil.sha256Hex("Notch"), HashUtil.sha256Hex("notch"));
    }

    @Test
    public void sha256Hex_anyInput_expectsSixtyFourLowercaseHexChars() {
        String digest = HashUtil.sha256Hex("SomePlayer123");
        assertEquals(64, digest.length());
        assertTrue("digest must be lowercase hex", digest.matches("[0-9a-f]{64}"));
    }

    @Test
    public void sha256Hex_nullInput_expectsIllegalArgumentException() {
        try {
            HashUtil.sha256Hex(null);
            fail("expected IllegalArgumentException for null input");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void sha256Hex_unicodeUsername_expectsCorrectUtf8BasedDigest() {
        // Usernames with non-ASCII characters must be hashed via their UTF-8 bytes, not
        // truncated or mangled; verified independently via Python's hashlib.
        assertEquals("fd181ee786a4492ddaab65578913673a1faa90cfec7b043604a1ec8e8f91f259",
                HashUtil.sha256Hex("Player_éè中文"));
    }
}
