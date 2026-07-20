package org.infinispan.tutorial.simple.hr;

import org.infinispan.api.annotations.indexing.Keyword;
import org.infinispan.protostream.annotations.Proto;

@Proto
public record ProjectAssignment(
      @Keyword String projectName,
      @Keyword String projectType
) {}
