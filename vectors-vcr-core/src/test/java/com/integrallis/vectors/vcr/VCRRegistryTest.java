package com.integrallis.vectors.vcr;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VCRRegistryTest {

  @Test
  void unknownTestsAreMissing() {
    VCRRegistry reg = new VCRRegistry(new HeapStorageBackend());
    assertEquals(VCRRegistry.RecordingStatus.MISSING, reg.getTestStatus("T:x"));
  }

  @Test
  void successOverridesPreviousFailure() {
    VCRRegistry reg = new VCRRegistry(new HeapStorageBackend());
    reg.registerFailure("T:x");
    reg.registerSuccess("T:x");
    assertEquals(VCRRegistry.RecordingStatus.RECORDED, reg.getTestStatus("T:x"));
  }

  @Test
  void recordNewUsesRecordForMissing() {
    VCRRegistry reg = new VCRRegistry(new HeapStorageBackend());
    assertEquals(VCRMode.RECORD, reg.determineEffectiveMode("T:new", VCRMode.RECORD_NEW));
    reg.registerSuccess("T:new");
    assertEquals(VCRMode.PLAYBACK, reg.determineEffectiveMode("T:new", VCRMode.RECORD_NEW));
  }

  @Test
  void recordFailedReRecordsOnlyFailed() {
    VCRRegistry reg = new VCRRegistry(new HeapStorageBackend());
    reg.registerSuccess("T:ok");
    reg.registerFailure("T:bad");
    assertEquals(VCRMode.PLAYBACK, reg.determineEffectiveMode("T:ok", VCRMode.RECORD_FAILED));
    assertEquals(VCRMode.RECORD, reg.determineEffectiveMode("T:bad", VCRMode.RECORD_FAILED));
  }

  @Test
  void playbackOrRecordRoutesOnStatus() {
    VCRRegistry reg = new VCRRegistry(new HeapStorageBackend());
    assertEquals(VCRMode.RECORD, reg.determineEffectiveMode("T:m", VCRMode.PLAYBACK_OR_RECORD));
    reg.registerSuccess("T:m");
    assertEquals(VCRMode.PLAYBACK, reg.determineEffectiveMode("T:m", VCRMode.PLAYBACK_OR_RECORD));
  }

  @Test
  void fixedModesArePassedThrough() {
    VCRRegistry reg = new VCRRegistry(new HeapStorageBackend());
    assertEquals(VCRMode.PLAYBACK, reg.determineEffectiveMode("T:x", VCRMode.PLAYBACK));
    assertEquals(VCRMode.RECORD, reg.determineEffectiveMode("T:x", VCRMode.RECORD));
    assertEquals(VCRMode.OFF, reg.determineEffectiveMode("T:x", VCRMode.OFF));
  }
}
