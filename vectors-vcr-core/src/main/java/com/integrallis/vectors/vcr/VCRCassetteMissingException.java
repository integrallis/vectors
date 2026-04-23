package com.integrallis.vectors.vcr;

/**
 * Exception thrown when a VCR cassette is not found during playback mode.
 *
 * <p>This exception indicates that the test expected to find a recorded cassette but none was
 * available. To fix this, run the test in RECORD or PLAYBACK_OR_RECORD mode first.
 */
public class VCRCassetteMissingException extends RuntimeException {

  private static final long serialVersionUID = 1L;

  private final String cassetteKey;
  private final String testId;

  /**
   * Creates a new exception.
   *
   * @param cassetteKey the key that was not found
   * @param testId the test identifier
   */
  public VCRCassetteMissingException(String cassetteKey, String testId) {
    super(
        String.format(
            "VCR cassette not found for test '%s'%nCassette key: %s%n"
                + "Run with VCRMode.RECORD or VCRMode.PLAYBACK_OR_RECORD to record this interaction",
            testId, cassetteKey));
    this.cassetteKey = cassetteKey;
    this.testId = testId;
  }

  /**
   * Gets the cassette key that was not found.
   *
   * @return the cassette key
   */
  public String getCassetteKey() {
    return cassetteKey;
  }

  /**
   * Gets the test identifier.
   *
   * @return the test ID
   */
  public String getTestId() {
    return testId;
  }
}
