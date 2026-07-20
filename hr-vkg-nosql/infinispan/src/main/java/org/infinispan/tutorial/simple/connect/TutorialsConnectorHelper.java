package org.infinispan.tutorial.simple.connect;

import org.infinispan.client.hotrod.RemoteCache;
import org.infinispan.client.hotrod.RemoteCacheManager;
import org.infinispan.client.hotrod.configuration.ClientIntelligence;
import org.infinispan.client.hotrod.configuration.ConfigurationBuilder;
import org.infinispan.commons.util.OS;

public class TutorialsConnectorHelper {

   public static final String USER = "admin";
   public static final String PASSWORD = "password";

   public static final ConfigurationBuilder connectionConfig() {
      ConfigurationBuilder builder = new ConfigurationBuilder();
      builder.security()
            .authentication()
            .username(USER)
            .password(PASSWORD);

      if (OS.getCurrentOs().equals(OS.MAC_OS) || OS.getCurrentOs().equals(OS.WINDOWS)) {
         builder.clientIntelligence(ClientIntelligence.BASIC);
      }

      return builder;
   }

   public static final RemoteCacheManager connect(ConfigurationBuilder builder) {
      RemoteCacheManager cacheManager;
      try {
         cacheManager = new RemoteCacheManager(builder.build());
         System.out.println("Connected! Cache names: " + cacheManager.getCacheNames());
      } catch (Exception ex) {
         System.out.println("Unable to connect to Infinispan Server at localhost:11222");
         System.out.println("Start the server with: docker run -p 11222:11222 -e USER=\"admin\" -e PASS=\"password\" quay.io/infinispan/server:16.2.1");
         throw ex;
      }
      return cacheManager;
   }

   public static void stop(RemoteCacheManager cacheManager) {
      if (cacheManager != null) {
         cacheManager.stop();
      }
   }
}
