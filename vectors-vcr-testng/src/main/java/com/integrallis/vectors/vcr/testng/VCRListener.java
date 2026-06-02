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
package com.integrallis.vectors.vcr.testng;

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
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.testng.IInvokedMethod;
import org.testng.IInvokedMethodListener;
import org.testng.ITestContext;
import org.testng.ITestListener;
import org.testng.ITestResult;

/**
 * TestNG listener that mirrors {@code VCRExtension} from the JUnit 5 module.
 *
 * <p>Opens a {@link VCRContext} per test class (keyed by annotated class), wraps {@code @VCRModel}
 * fields before each test method, and records success/failure on the {@link VCRRegistry}.
 */
public final class VCRListener implements ITestListener, IInvokedMethodListener {

  /** System-property key that overrides {@link VCRTestNG#dataDir()} at runtime. */
  public static final String DATA_DIR_SYSPROP = "vcr.dataDir";

  private final ConcurrentMap<Class<?>, VCRContext> contexts = new ConcurrentHashMap<>();

  @Override
  public void onStart(ITestContext context) {
    for (Class<?> cls : distinctClasses(context)) {
      VCRTestNG config = cls.getAnnotation(VCRTestNG.class);
      if (config == null) {
        continue;
      }
      try {
        contexts.computeIfAbsent(cls, c -> buildContext(config));
      } catch (RuntimeException e) {
        throw new RuntimeException("Failed to initialise VCR for " + cls.getName(), e);
      }
    }
  }

  @Override
  public void beforeInvocation(IInvokedMethod method, ITestResult result) {
    if (!method.isTestMethod()) {
      return;
    }
    Class<?> testClass = result.getTestClass().getRealClass();
    VCRContext vcrContext = contexts.get(testClass);
    if (vcrContext == null) {
      return;
    }
    String testId = testId(result);
    vcrContext.resetCallCounters();
    vcrContext.setCurrentTest(testId);

    Method javaMethod = result.getMethod().getConstructorOrMethod().getMethod();
    VCRMode effectiveMode;
    if (javaMethod != null && javaMethod.isAnnotationPresent(VCRDisabled.class)) {
      effectiveMode = VCRMode.OFF;
    } else if (javaMethod != null && javaMethod.isAnnotationPresent(VCRRecord.class)) {
      effectiveMode = VCRMode.RECORD;
    } else {
      effectiveMode =
          vcrContext.getRegistry().determineEffectiveMode(testId, vcrContext.getEffectiveMode());
    }
    vcrContext.setEffectiveMode(effectiveMode);
    if (effectiveMode == VCRMode.OFF) {
      return;
    }
    wrapFields(result.getInstance(), testId, effectiveMode, vcrContext.getCassetteStore());
  }

  @Override
  public void onTestSuccess(ITestResult result) {
    VCRContext vcrContext = contexts.get(result.getTestClass().getRealClass());
    if (vcrContext != null && vcrContext.getEffectiveMode().isRecordMode()) {
      vcrContext.getRegistry().registerSuccess(testId(result));
    }
  }

  @Override
  public void onTestFailure(ITestResult result) {
    VCRContext vcrContext = contexts.get(result.getTestClass().getRealClass());
    if (vcrContext == null) {
      return;
    }
    if (vcrContext.getEffectiveMode().isRecordMode()) {
      vcrContext.getRegistry().registerFailure(testId(result));
      for (var key : vcrContext.getCurrentCassetteKeys()) {
        vcrContext.getCassetteStore().delete(key);
      }
    }
  }

  @Override
  public void onFinish(ITestContext context) {
    IOException failure = null;
    for (VCRContext vcrContext : contexts.values()) {
      try {
        flushAndClose(vcrContext.getCassetteStore());
      } catch (IOException e) {
        if (failure == null) {
          failure = e;
        } else {
          failure.addSuppressed(e);
        }
      }
    }
    contexts.clear();
    if (failure != null) {
      throw new RuntimeException("VCR cassette flush/close failed", failure);
    }
  }

  static void flushAndClose(CassetteStore store) throws IOException {
    IOException failure = null;
    try {
      store.flush();
    } catch (IOException e) {
      failure = e;
    }
    try {
      store.close();
    } catch (IOException e) {
      if (failure == null) {
        failure = e;
      } else {
        failure.addSuppressed(e);
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  private static VCRContext buildContext(VCRTestNG config) {
    try {
      String override = System.getProperty(DATA_DIR_SYSPROP);
      String dataDirStr = (override != null && !override.isEmpty()) ? override : config.dataDir();
      Path dataDir = Path.of(dataDirStr).toAbsolutePath();
      Files.createDirectories(dataDir);
      CassetteStore store = new ExactCassetteStore(new LocalFileStorageBackend(dataDir));
      VCRRegistry registry = new VCRRegistry(new LocalFileStorageBackend(dataDir));
      return new VCRContext(store, registry, config.mode());
    } catch (IOException e) {
      throw new RuntimeException("Failed to create VCR data directory", e);
    }
  }

  private static String testId(ITestResult result) {
    return result.getTestClass().getRealClass().getName()
        + ":"
        + result.getMethod().getMethodName();
  }

  private static java.util.Set<Class<?>> distinctClasses(ITestContext context) {
    java.util.Set<Class<?>> classes = new java.util.LinkedHashSet<>();
    for (var test : context.getAllTestMethods()) {
      classes.add(test.getRealClass());
    }
    return classes;
  }

  private static void wrapFields(
      Object instance, String testId, VCRMode mode, CassetteStore store) {
    if (instance == null) {
      return;
    }
    Class<?> current = instance.getClass();
    while (current != null && current != Object.class) {
      for (Field field : current.getDeclaredFields()) {
        VCRModel ann = field.getAnnotation(VCRModel.class);
        if (ann != null) {
          VCRModelWrapper.wrapField(instance, field, testId, mode, ann.modelName(), store);
        }
      }
      current = current.getSuperclass();
    }
  }
}
