package org.infinispan.tutorial.simple.hr;

import java.util.List;

import org.infinispan.api.annotations.indexing.Basic;
import org.infinispan.api.annotations.indexing.Embedded;
import org.infinispan.api.annotations.indexing.Indexed;
import org.infinispan.api.annotations.indexing.Keyword;
import org.infinispan.api.annotations.indexing.Text;
import org.infinispan.api.annotations.indexing.option.Structure;
import org.infinispan.protostream.annotations.Proto;

@Proto
@Indexed
public record EmployeeProfile(
      @Keyword(projectable = true, sortable = true, normalizer = "lowercase") String personName,
      @Keyword(projectable = true, sortable = true, normalizer = "lowercase") String surname,
      @Keyword String email,
      @Basic Integer employeeId,
      @Keyword(projectable = true, sortable = true) String role,
      @Keyword(projectable = true, sortable = true) String departmentCode,
      @Text String address,
      @Keyword String citizenStatus,
      @Keyword String maritalStatus,
      @Embedded(structure = Structure.NESTED) List<Contract> contracts,
      @Embedded(structure = Structure.NESTED) List<SkillLevel> skills,
      @Embedded(structure = Structure.NESTED) List<ProjectAssignment> projects
) {}
