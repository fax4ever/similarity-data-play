# Human Resource with Virtual Knowledge Graph and NoSQL

> a Big Data/Mgmt project

## Spec

1. Define an OWL 2 QL Ontology (to support first-order query rewritability) on Human Resource domain (using Protégé)
2. Define a SQL schema to be imported using Ontop Protégé plugin with a virtual knowledge graph technique
   * In particular, define at least 3 different mapping patterns (of the ones described in VKGs slides)
3. Define at least 5 SPARQL queries to query the ontology data
4. Design the same ontology using a key/value (partially document) NoSQL [Infinispan](https://infinispan.org/)
   * Annotate the Infinispan POJOs to add Apache Lucene index to support document-like queries
   * Indexing the values of the caches allows us to treat them as documents (so we can query the inner fields)
5. Define at least 5 Infinispan (called Ickle) queries using the Lucene indexes

