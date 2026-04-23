/**
 * TestNG integration for VCR record/playback. Mirrors the {@code vectors-vcr-junit5} module API.
 *
 * <p>Annotate a TestNG test class with {@link com.integrallis.vectors.vcr.testng.VCRTestNG} and
 * register {@link com.integrallis.vectors.vcr.testng.VCRListener} as a TestNG listener (already
 * triggered automatically via {@link org.testng.annotations.Listeners} on the annotation).
 */
package com.integrallis.vectors.vcr.testng;
