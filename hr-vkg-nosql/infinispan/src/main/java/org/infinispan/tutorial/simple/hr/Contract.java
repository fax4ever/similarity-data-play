package org.infinispan.tutorial.simple.hr;

import org.infinispan.api.annotations.indexing.Basic;
import org.infinispan.api.annotations.indexing.Keyword;
import org.infinispan.protostream.annotations.Proto;

@Proto
public record Contract(
      @Basic Integer contractNum,
      @Keyword(sortable = true) String startDate,
      @Keyword(sortable = true) String endDate,
      @Basic(sortable = true) Double salary
) {}
