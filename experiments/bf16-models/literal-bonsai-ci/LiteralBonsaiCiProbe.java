package com.integrallis.models.backend.nativekernel;
public class LiteralBonsaiCiProbe { public static void main(String[] a) throws Exception { var test = new Bonsai2NativeModelFixtureSlowTest(); test.transformedBatchesMatchSequentialRealWeightLogitsAcrossChunkBoundaries(); System.out.println("LITERAL_BONSAI_CI_PASS"); } }
