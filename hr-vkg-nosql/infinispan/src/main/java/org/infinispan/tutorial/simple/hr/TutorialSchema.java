package org.infinispan.tutorial.simple.hr;

import org.infinispan.protostream.GeneratedSchema;
import org.infinispan.protostream.annotations.ProtoSchema;

@ProtoSchema(schemaPackageName = "tutorial",
      includeClasses = { EmployeeProfile.class, Contract.class, SkillLevel.class, ProjectAssignment.class })
public interface TutorialSchema extends GeneratedSchema {
}
