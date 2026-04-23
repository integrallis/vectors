package com.integrallis.vectors.vcr;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Forces a specific test method to use RECORD mode, overriding the class-level VCR mode.
 *
 * <p>When applied to a test method, this annotation forces that specific test to always make real
 * API calls and record the responses, regardless of the class-level VCR mode setting.
 *
 * @see VCRDisabled
 * @see VCRMode#RECORD
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface VCRRecord {}
