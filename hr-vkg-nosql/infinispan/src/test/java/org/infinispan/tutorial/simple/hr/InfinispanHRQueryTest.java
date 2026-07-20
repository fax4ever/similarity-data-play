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
            "FROM tutorial.EmployeeProfile ORDER BY e.departmentCode, e.surname");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(10);
   }

   @Test
   public void queryByDepartmentWithProjection() {
      Query<Object[]> query = InfinispanHRQuery.cache.query(
            "SELECT e.personName, e.surname, e.role " +
            "FROM tutorial.EmployeeProfile e " +
            "WHERE e.departmentCode = :dept ORDER BY e.surname");
      query.setParameter("dept", "HR");
      List<Object[]> result = query.execute().list();

      assertThat(result).hasSize(3);
      Set<String> surnames = result.stream().map(r -> (String) r[1]).collect(Collectors.toSet());
      assertThat(surnames).containsExactlyInAnyOrder("Rossi", "Verdi", "Colombo");
   }

   @Test
   public void queryBySkillAndLevel() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e " +
            "WHERE e.skills.skillName = :skill AND e.skills.level = :level");
      query.setParameter("skill", "Java");
      query.setParameter("level", "expert");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(1);
      assertThat(result.get(0).surname()).isEqualTo("Neri");
   }

   @Test
   public void queryManagersOnSoftwareProjects() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e " +
            "WHERE e.role IN ('manager', 'director') AND e.projects.projectType = :type");
      query.setParameter("type", "software");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(1);
      assertThat(result.get(0).surname()).isEqualTo("Neri");
   }

   @Test
   public void queryHighSalaryByDepartment() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e " +
            "WHERE e.contracts.salary > :minSalary AND e.departmentCode = :dept " +
            "ORDER BY e.surname");
      query.setParameter("minSalary", 50000.0);
      query.setParameter("dept", "HR");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSizeGreaterThanOrEqualTo(2);
      Set<String> surnames = result.stream().map(EmployeeProfile::surname).collect(Collectors.toSet());
      assertThat(surnames).contains("Rossi", "Verdi");
   }

   @Test
   public void queryFullTextAddressWithSkillCategory() {
      Query<EmployeeProfile> query = InfinispanHRQuery.cache.query(
            "FROM tutorial.EmployeeProfile e " +
            "WHERE e.address : 'Milan' AND e.skills.category = :cat");
      query.setParameter("cat", "Programming");
      List<EmployeeProfile> result = query.execute().list();

      assertThat(result).hasSize(3);
      Set<String> surnames = result.stream().map(EmployeeProfile::surname).collect(Collectors.toSet());
      assertThat(surnames).containsExactlyInAnyOrder("Neri", "Colombo", "Ferrari");
   }

   @Test
   public void queryHeadcountByDepartment() {
      Query<Object[]> query = InfinispanHRQuery.cache.query(
            "SELECT e.departmentCode, COUNT(e.departmentCode) " +
            "FROM tutorial.EmployeeProfile e " +
            "GROUP BY e.departmentCode " +
            "ORDER BY COUNT(e.departmentCode) DESC");
      List<Object[]> result = query.execute().list();

      assertThat(result).hasSize(5);
      assertThat(result.get(0)[0]).isEqualTo("HR");
      assertThat((Long) result.get(0)[1]).isEqualTo(3);
   }
}
