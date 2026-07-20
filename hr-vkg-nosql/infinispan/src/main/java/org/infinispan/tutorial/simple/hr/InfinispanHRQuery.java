package org.infinispan.tutorial.simple.hr;

import org.infinispan.client.hotrod.RemoteCache;
import org.infinispan.client.hotrod.RemoteCacheManager;
import org.infinispan.client.hotrod.configuration.ConfigurationBuilder;
import org.infinispan.protostream.GeneratedSchema;
import org.infinispan.tutorial.simple.connect.TutorialsConnectorHelper;

import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.infinispan.query.remote.client.ProtobufMetadataManagerConstants.PROTOBUF_METADATA_CACHE_NAME;

public class InfinispanHRQuery {

   public static final String CACHE_NAME = "employeeProfiles";
   static RemoteCacheManager client;
   static RemoteCache<String, EmployeeProfile> cache;

   static void connectToInfinispan() throws Exception {
      ConfigurationBuilder builder = TutorialsConnectorHelper.connectionConfig();

      builder.addContextInitializer(new TutorialSchemaImpl());

      URI indexedCacheURI = InfinispanHRQuery.class.getClassLoader()
            .getResource("indexedCache.xml").toURI();
      builder.remoteCache(CACHE_NAME).configurationURI(indexedCacheURI);

      client = TutorialsConnectorHelper.connect(builder);

      addSchema(client);

      cache = client.getCache(CACHE_NAME);
      cache.clear();
   }

   static void addDataToCache() {
      Map<String, EmployeeProfile> employees = new HashMap<>();

      employees.put("person-1", new EmployeeProfile(
            "Alice", "Rossi", "alice.rossi@company.com", 1001, "director", "HR",
            "10 Via Roma, Milan", "citizen", "married",
            List.of(
                  new Contract(1, "2005-01-01", "2010-12-31", 55000.0),
                  new Contract(2, "2011-01-01", "2018-12-31", 72000.0),
                  new Contract(3, "2019-01-01", null, 95000.0)),
            List.of(
                  new SkillLevel("Project Management", "Management", "expert"),
                  new SkillLevel("Data Analysis", "Analytics", "advanced"),
                  new SkillLevel("Cloud Architecture", "Infrastructure", "intermediate")),
            List.of(
                  new ProjectAssignment("Talent Acquisition Platform", "hr"),
                  new ProjectAssignment("Employee Wellness Program", "hr"))));

      employees.put("person-2", new EmployeeProfile(
            "Bob", "Bianchi", "bob.bianchi@company.com", 1002, "director", "FIN",
            "25 Corso Buenos Aires, Milan", "citizen", "married",
            List.of(
                  new Contract(1, "2006-03-01", "2012-02-28", 52000.0),
                  new Contract(2, "2012-03-01", null, 88000.0)),
            List.of(
                  new SkillLevel("SQL", "Database", "expert"),
                  new SkillLevel("Project Management", "Management", "expert"),
                  new SkillLevel("Data Analysis", "Analytics", "advanced")),
            List.of(
                  new ProjectAssignment("Supply Chain Optimization", "business"))));

      employees.put("person-3", new EmployeeProfile(
            "Clara", "Verdi", "clara.verdi@company.com", 1003, "manager", "HR",
            "8 Via Torino, Milan", "citizen", "single",
            List.of(
                  new Contract(1, "2010-06-01", "2016-05-31", 45000.0),
                  new Contract(2, "2016-06-01", null, 62000.0)),
            List.of(
                  new SkillLevel("Project Management", "Management", "advanced"),
                  new SkillLevel("UX Design", "Design", "intermediate")),
            List.of(
                  new ProjectAssignment("Talent Acquisition Platform", "hr"),
                  new ProjectAssignment("Employee Wellness Program", "hr"))));

      employees.put("person-4", new EmployeeProfile(
            "David", "Neri", "david.neri@company.com", 1004, "manager", "ENG",
            "42 Viale Monza, Milan", "citizen", "married",
            List.of(
                  new Contract(1, "2008-09-01", "2015-08-31", 48000.0),
                  new Contract(2, "2015-09-01", null, 67000.0)),
            List.of(
                  new SkillLevel("Java", "Programming", "expert"),
                  new SkillLevel("Python", "Programming", "advanced"),
                  new SkillLevel("Cloud Architecture", "Infrastructure", "expert")),
            List.of(
                  new ProjectAssignment("Core Banking Rewrite", "software"),
                  new ProjectAssignment("Mobile App v3", "software"))));

      employees.put("person-5", new EmployeeProfile(
            "Elena", "Russo", "elena.russo@company.com", 1005, "manager", "MKT",
            "15 Via Dante, Milan", "citizen", "single",
            List.of(
                  new Contract(1, "2015-02-01", null, 58000.0)),
            List.of(
                  new SkillLevel("Data Analysis", "Analytics", "advanced"),
                  new SkillLevel("UX Design", "Design", "expert"),
                  new SkillLevel("Machine Learning", "Analytics", "intermediate")),
            List.of(
                  new ProjectAssignment("Market Expansion APAC", "business"))));

      employees.put("person-6", new EmployeeProfile(
            "Frank", "Colombo", "frank.colombo@company.com", 1006, "employee", "HR",
            "3 Via Manzoni, Milan", "citizen", "single",
            List.of(
                  new Contract(1, "2018-04-01", null, 42000.0)),
            List.of(
                  new SkillLevel("Java", "Programming", "intermediate"),
                  new SkillLevel("SQL", "Database", "beginner")),
            List.of(
                  new ProjectAssignment("Talent Acquisition Platform", "hr"),
                  new ProjectAssignment("Employee Wellness Program", "hr"))));

      employees.put("person-7", new EmployeeProfile(
            "Giulia", "Ferrari", "giulia.ferrari@company.com", 1007, "employee", "ENG",
            "77 Corso Venezia, Milan", "citizen", "married",
            List.of(
                  new Contract(1, "2014-07-01", "2019-06-30", 44000.0),
                  new Contract(2, "2019-07-01", null, 53000.0)),
            List.of(
                  new SkillLevel("Java", "Programming", "advanced"),
                  new SkillLevel("Python", "Programming", "expert"),
                  new SkillLevel("Cloud Architecture", "Infrastructure", "advanced")),
            List.of(
                  new ProjectAssignment("Core Banking Rewrite", "software"),
                  new ProjectAssignment("Mobile App v3", "software"))));

      employees.put("person-8", new EmployeeProfile(
            "Hans", "Muller", "hans.muller@company.com", 1008, "employee", "FIN",
            "18 Via Garibaldi, Bergamo", "resident", "single",
            List.of(
                  new Contract(1, "2017-01-01", null, 46000.0)),
            List.of(
                  new SkillLevel("SQL", "Database", "advanced"),
                  new SkillLevel("Data Analysis", "Analytics", "intermediate")),
            List.of(
                  new ProjectAssignment("Supply Chain Optimization", "business"))));

      employees.put("person-9", new EmployeeProfile(
            "Irene", "Esposito", "irene.esposito@company.com", 1009, "employee", "MKT",
            "5 Via Verdi, Turin", "citizen", "married",
            List.of(
                  new Contract(1, "2019-09-01", null, 41000.0)),
            List.of(
                  new SkillLevel("UX Design", "Design", "advanced"),
                  new SkillLevel("Machine Learning", "Analytics", "beginner")),
            List.of(
                  new ProjectAssignment("Market Expansion APAC", "business"))));

      employees.put("person-10", new EmployeeProfile(
            "James", "Romano", "james.romano@company.com", 1010, "employee", "OPS",
            "22 Via Pascoli, Rome", "citizen", "single",
            List.of(
                  new Contract(1, "2013-11-01", "2018-10-31", 39000.0),
                  new Contract(2, "2018-11-01", null, 47000.0)),
            List.of(
                  new SkillLevel("Python", "Programming", "intermediate"),
                  new SkillLevel("Cloud Architecture", "Infrastructure", "beginner")),
            List.of(
                  new ProjectAssignment("Core Banking Rewrite", "software"))));

      cache.putAll(employees);
   }

   public static void disconnect(boolean removeCaches) {
      if (removeCaches) {
         client.administration().removeCache(CACHE_NAME);
      }
      TutorialsConnectorHelper.stop(client);
   }

   private static void addSchema(RemoteCacheManager cacheManager) {
      RemoteCache<String, String> metadataCache =
            cacheManager.getCache(PROTOBUF_METADATA_CACHE_NAME);
      GeneratedSchema schema = new TutorialSchemaImpl();
      metadataCache.put(schema.getProtoFileName(), schema.getProtoFile());
   }
}
