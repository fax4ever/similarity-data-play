package org.infinispan.tutorial.simple.hr;

import static org.assertj.core.api.Assertions.assertThat;

import org.infinispan.commons.api.query.Query;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class InfinispanHRQueryTest {

   @BeforeAll
   public void start() throws Exception {
      InfinispanHRQuery.connectToInfinispan();
      InfinispanHRQuery.addDataToCache();
   }

   @AfterAll
   public void stop() {
      InfinispanHRQuery.disconnect(true);
   }

   @Test
   public void queryAllEmployees() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(10);
   }

   @Test
   public void queryByDepartment() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e WHERE e.departmentCode = :dept ORDER BY e.surname");
      query.setParameter("dept", "ENG");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(2);
      Set<String> surnames = result.stream().map(EmployeeProfile::surname).collect(Collectors.toSet());
      assertThat(surnames).containsExactlyInAnyOrder("Neri", "Ferrari");
   }

   @Test
   public void queryBySkill() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e WHERE e.skills.skillName = :skill");
      query.setParameter("skill", "Java");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(3);
      Set<String> surnames = result.stream().map(EmployeeProfile::surname).collect(Collectors.toSet());
      assertThat(surnames).containsExactlyInAnyOrder("Neri", "Colombo", "Ferrari");
   }

   @Test
   public void queryByProject() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e WHERE e.projects.projectName = :project");
      query.setParameter("project", "Core Banking Rewrite");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(3);
      Set<String> surnames = result.stream().map(EmployeeProfile::surname).collect(Collectors.toSet());
      assertThat(surnames).containsExactlyInAnyOrder("Neri", "Ferrari", "Romano");
   }

   @Test
   public void queryHighSalary() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e WHERE e.contracts.salary > :minSalary");
      query.setParameter("minSalary", 60000.0);
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSizeGreaterThanOrEqualTo(4);
      Set<String> surnames = result.stream().map(EmployeeProfile::surname).collect(Collectors.toSet());
      assertThat(surnames).contains("Rossi", "Bianchi", "Verdi", "Neri");
   }

   @Test
   public void queryFullTextAddress() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e WHERE e.address : 'Milan'");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(8);
      result.forEach(e -> assertThat(e.address()).containsIgnoringCase("Milan"));
   }
}
