/**
 * JUnit 5 extension that automates VCR record/playback for LLM integration tests.
 *
 * <p>Annotate a test class with {@link com.integrallis.vectors.vcr.junit5.VCRTest} and it will be
 * automatically {@code @ExtendWith}-ed with {@link
 * com.integrallis.vectors.vcr.junit5.VCRExtension}. Annotated {@code @VCRModel} fields are wrapped
 * on each {@code @BeforeEach} via {@link com.integrallis.vectors.vcr.VCRModelWrapper}.
 */
package com.integrallis.vectors.vcr.junit5;
