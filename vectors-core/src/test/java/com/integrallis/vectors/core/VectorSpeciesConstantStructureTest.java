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
package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.FieldModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.classfile.constantpool.PoolEntry;
import java.lang.constant.ClassDesc;
import java.lang.reflect.AccessFlag;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;

/**
 * Structural rule: a {@code VectorSpecies} is a compile-time constant, never a value.
 *
 * <p>C2 only specialises a Vector API kernel when the species folds to a constant. A species that
 * arrives as a method parameter, or is read from an instance or non-final static field, reaches the
 * intrinsics as a runtime value unless the whole call chain happens to inline, and the kernel
 * silently falls back to boxed Java execution. The rule is checked on the compiled bytecode of
 * every vectors-core main class that references {@code jdk.incubator.vector}, so lambdas and
 * synthetic methods are covered too.
 *
 * <p>The detector itself is tested against fixtures below, so a green run cannot mean "the scan
 * never looked".
 */
class VectorSpeciesConstantStructureTest {

  private static final String SPECIES_DESCRIPTOR =
      VectorSpecies.class.describeConstable().orElseThrow().descriptorString();

  /**
   * Known violations present when this rule was introduced (2026-09-16, main 5ce9b4c). Each is a
   * follow-up, not a licence: remove the entry when the kernel is rewritten to use a static final
   * species. New entries need a comment explaining why the species must be a value.
   */
  private static final Set<String> ALLOW_LIST =
      Set.of(
          // Follow-up: generic capping helper runs once at class init to build the static final
          // species constants; it never executes on a hot path. Acceptable as-is.
          "com.integrallis.vectors.core.PanamaConstants#preferredSpecies("
              + "Ljdk/incubator/vector/VectorSpecies;)Ljdk/incubator/vector/VectorSpecies;");

  @Test
  void noVectorSpeciesParametersOrMutableSpeciesFieldsInMainClasses() throws Exception {
    Path classesRoot = mainClassesRoot();
    List<String> scanned = new ArrayList<>();
    Set<String> violations = new TreeSet<>();
    try (Stream<Path> files = Files.walk(classesRoot)) {
      for (Path file : files.filter(path -> path.toString().endsWith(".class")).toList()) {
        byte[] bytes = Files.readAllBytes(file);
        ClassModel model = ClassFile.of().parse(bytes);
        if (!usesVectorApi(model)) {
          continue;
        }
        scanned.add(model.thisClass().asInternalName());
        violations.addAll(violations(model));
      }
    }

    // Observability: the scan must actually reach the Panama kernels, or a green run is vacuous.
    assertThat(scanned)
        .as("classes referencing jdk.incubator.vector under %s", classesRoot)
        .contains("com/integrallis/vectors/core/PanamaVectorUtilSupport");

    Set<String> unexpected = new TreeSet<>(violations);
    unexpected.removeAll(ALLOW_LIST);
    assertThat(unexpected)
        .as("VectorSpecies must be a static final constant, never a parameter or mutable field")
        .isEmpty();

    Set<String> stale = new TreeSet<>(ALLOW_LIST);
    stale.removeAll(violations);
    assertThat(stale).as("allow-list entries that no longer violate; delete them").isEmpty();
  }

  @Test
  void detectorFlagsSpeciesParameter() {
    assertThat(violations(parse(SpeciesParameterFixture.class)))
        .singleElement()
        .asString()
        .contains("#load(" + SPECIES_DESCRIPTOR);
  }

  @Test
  void detectorFlagsInstanceAndNonFinalStaticSpeciesFields() {
    assertThat(violations(parse(SpeciesFieldFixture.class)))
        .containsExactlyInAnyOrder(
            SpeciesFieldFixture.class.getName() + "#instanceSpecies",
            SpeciesFieldFixture.class.getName() + "#mutableStaticSpecies");
  }

  @Test
  void detectorAcceptsStaticFinalSpecies() {
    ClassModel model = parse(StaticFinalSpeciesFixture.class);
    assertThat(usesVectorApi(model)).isTrue();
    assertThat(violations(model)).isEmpty();
  }

  static boolean usesVectorApi(ClassModel model) {
    for (PoolEntry entry : model.constantPool()) {
      if (entry instanceof ClassEntry classEntry
          && classEntry.asInternalName().startsWith("jdk/incubator/vector/")) {
        return true;
      }
    }
    return false;
  }

  static List<String> violations(ClassModel model) {
    String owner = model.thisClass().asInternalName().replace('/', '.');
    List<String> found = new ArrayList<>();
    for (MethodModel method : model.methods()) {
      for (ClassDesc parameter : method.methodTypeSymbol().parameterList()) {
        if (parameter.descriptorString().equals(SPECIES_DESCRIPTOR)) {
          found.add(owner + "#" + method.methodName().stringValue() + method.methodType());
          break;
        }
      }
    }
    for (FieldModel field : model.fields()) {
      if (!field.fieldType().stringValue().equals(SPECIES_DESCRIPTOR)) {
        continue;
      }
      Set<AccessFlag> flags = field.flags().flags();
      if (!(flags.contains(AccessFlag.STATIC) && flags.contains(AccessFlag.FINAL))) {
        found.add(owner + "#" + field.fieldName().stringValue());
      }
    }
    return found;
  }

  private static Path mainClassesRoot() throws URISyntaxException {
    Path location =
        Path.of(
            PanamaVectorUtilSupport.class
                .getProtectionDomain()
                .getCodeSource()
                .getLocation()
                .toURI());
    assertThat(Files.isDirectory(location))
        .as("vectors-core main classes must be a directory on the test classpath: %s", location)
        .isTrue();
    return location;
  }

  private static ClassModel parse(Class<?> type) {
    String resource = type.getName().substring(type.getPackageName().length() + 1) + ".class";
    try (InputStream in = type.getResourceAsStream(resource)) {
      assertThat(in).as(resource).isNotNull();
      return ClassFile.of().parse(in.readAllBytes());
    } catch (IOException exception) {
      throw new UncheckedIOException(exception);
    }
  }

  static final class SpeciesParameterFixture {
    private SpeciesParameterFixture() {}

    static FloatVector load(VectorSpecies<Float> species, float[] values) {
      return FloatVector.fromArray(species, values, 0);
    }
  }

  static final class SpeciesFieldFixture {
    static VectorSpecies<Float> mutableStaticSpecies = FloatVector.SPECIES_PREFERRED;
    final VectorSpecies<Float> instanceSpecies = FloatVector.SPECIES_PREFERRED;
  }

  static final class StaticFinalSpeciesFixture {
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private StaticFinalSpeciesFixture() {}

    static float first(float[] values) {
      return FloatVector.fromArray(SPECIES, values, 0).lane(0);
    }
  }
}
