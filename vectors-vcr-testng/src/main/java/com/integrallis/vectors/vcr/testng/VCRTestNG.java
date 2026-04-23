package com.integrallis.vectors.vcr.testng;

import com.integrallis.vectors.vcr.VCRMode;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Enables VCR recording/playback for a TestNG test class.
 *
 * <p>This is the TestNG analog of {@code @VCRTest} in {@code vectors-vcr-junit5}. {@link
 * VCRListener} is automatically registered for every TestNG run through the {@code
 * META-INF/services/org.testng.ITestNGListener} file shipped with this module; the listener is a
 * no-op for classes that do not carry {@code @VCRTestNG}.
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface VCRTestNG {

  /**
   * @return the default VCR mode for the annotated class
   */
  VCRMode mode() default VCRMode.PLAYBACK_OR_RECORD;

  /**
   * @return the directory under which cassettes are persisted
   */
  String dataDir() default "src/test/resources/vcr-data";
}
