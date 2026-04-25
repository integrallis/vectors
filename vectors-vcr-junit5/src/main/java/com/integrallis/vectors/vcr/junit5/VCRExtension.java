/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.vcr.junit5;

import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRContext;
import com.integrallis.vectors.vcr.VCRDisabled;
import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRModel;
import com.integrallis.vectors.vcr.VCRModelWrapper;
import com.integrallis.vectors.vcr.VCRRecord;
import com.integrallis.vectors.vcr.VCRRegistry;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;
import org.junit.jupiter.api.extension.AfterAllCallback;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.BeforeTestExecutionCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.TestWatcher;

/**
 * JUnit 5 extension that drives the VCR lifecycle for a {@link VCRTest}-annotated class.
 *
 * <p>Per session: creates a {@link CassetteStore} (either from a registered {@link
 * CassetteStoreFactory} or a file-backed default), wraps {@code @VCRModel} fields on each test, and
 * updates the {@link VCRRegistry} on success/failure.
 */
public final class VCRExtension
    implements BeforeAllCallback, AfterAllCallback, BeforeTestExecutionCallback, TestWatcher {

  private static final ExtensionContext.Namespace NAMESPACE =
      ExtensionContext.Namespace.create(VCRExtension.class);
  private static final String CTX_KEY = "vcr-context";

  /** System-property key that overrides {@link VCRTest#dataDir()} at runtime (test-only hook). */
  public static final String DATA_DIR_SYSPROP = "vcr.dataDir";

  @Override
  public void beforeAll(ExtensionContext ctx) throws Exception {
    VCRTest config = findAnnotation(ctx.getRequiredTestClass());
    if (config == null) {
      config = DefaultVCRTest.INSTANCE;
    }
    String override = System.getProperty(DATA_DIR_SYSPROP);
    String dataDirStr = (override != null && !override.isEmpty()) ? override : config.dataDir();
    Path dataDir = Path.of(dataDirStr).toAbsolutePath();
    Files.createDirectories(dataDir);

    CassetteStore store = buildStore(dataDir);
    VCRRegistry registry = new VCRRegistry(new LocalFileStorageBackend(dataDir));
    VCRContext vcrContext = new VCRContext(store, registry, config.mode());
    ctx.getStore(NAMESPACE).put(CTX_KEY, vcrContext);
  }

  @Override
  public void beforeTestExecution(ExtensionContext ctx) {
    VCRContext vcrContext = getContext(ctx);
    if (vcrContext == null) {
      return;
    }
    String testId = testId(ctx);
    vcrContext.resetCallCounters();
    vcrContext.setCurrentTest(testId);

    var method = ctx.getRequiredTestMethod();
    VCRMode effectiveMode;
    if (method.isAnnotationPresent(VCRDisabled.class)) {
      effectiveMode = VCRMode.OFF;
    } else if (method.isAnnotationPresent(VCRRecord.class)) {
      effectiveMode = VCRMode.RECORD;
    } else {
      effectiveMode =
          vcrContext.getRegistry().determineEffectiveMode(testId, vcrContext.getEffectiveMode());
    }
    vcrContext.setEffectiveMode(effectiveMode);

    if (effectiveMode == VCRMode.OFF) {
      return;
    }
    wrapAnnotatedFields(
        ctx.getRequiredTestInstance(), testId, effectiveMode, vcrContext.getCassetteStore());
  }

  @Override
  public void testSuccessful(ExtensionContext ctx) {
    VCRContext vcrContext = getContext(ctx);
    if (vcrContext != null && vcrContext.getEffectiveMode().isRecordMode()) {
      vcrContext.getRegistry().registerSuccess(testId(ctx));
    }
  }

  @Override
  public void testFailed(ExtensionContext ctx, Throwable cause) {
    VCRContext vcrContext = getContext(ctx);
    if (vcrContext == null) {
      return;
    }
    if (vcrContext.getEffectiveMode().isRecordMode()) {
      vcrContext.getRegistry().registerFailure(testId(ctx));
      for (var key : vcrContext.getCurrentCassetteKeys()) {
        vcrContext.getCassetteStore().delete(key);
      }
    }
  }

  @Override
  public void afterAll(ExtensionContext ctx) throws Exception {
    VCRContext vcrContext = getContext(ctx);
    if (vcrContext != null) {
      try {
        vcrContext.getCassetteStore().flush();
      } catch (IOException e) {
        // Best-effort flush; do not fail the suite teardown on I/O errors here.
      }
      vcrContext.getCassetteStore().close();
    }
  }

  private static VCRContext getContext(ExtensionContext ctx) {
    return ctx.getStore(NAMESPACE).get(CTX_KEY, VCRContext.class);
  }

  private static String testId(ExtensionContext ctx) {
    return ctx.getRequiredTestClass().getName() + ":" + ctx.getRequiredTestMethod().getName();
  }

  private static VCRTest findAnnotation(Class<?> testClass) {
    Class<?> current = testClass;
    while (current != null) {
      VCRTest annotation = current.getAnnotation(VCRTest.class);
      if (annotation != null) {
        return annotation;
      }
      current = current.getEnclosingClass();
    }
    return null;
  }

  private static CassetteStore buildStore(Path dataDir) throws IOException {
    for (CassetteStoreFactory factory : ServiceLoader.load(CassetteStoreFactory.class)) {
      CassetteStore store = factory.create(dataDir);
      if (store != null) {
        return store;
      }
    }
    return new ExactCassetteStore(new LocalFileStorageBackend(dataDir));
  }

  private static void wrapAnnotatedFields(
      Object testInstance, String testId, VCRMode mode, CassetteStore store) {
    List<Field> fields = new ArrayList<>();
    Class<?> current = testInstance.getClass();
    while (current != null && current != Object.class) {
      for (Field f : current.getDeclaredFields()) {
        fields.add(f);
      }
      current = current.getSuperclass();
    }
    for (Field field : fields) {
      VCRModel annotation = field.getAnnotation(VCRModel.class);
      if (annotation != null) {
        VCRModelWrapper.wrapField(testInstance, field, testId, mode, annotation.modelName(), store);
      }
    }
  }

  @VCRTest
  private static class DefaultVCRTest {
    static final VCRTest INSTANCE = DefaultVCRTest.class.getAnnotation(VCRTest.class);
  }
}
