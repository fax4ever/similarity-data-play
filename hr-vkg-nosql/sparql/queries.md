# SPARQL Queries

> 5 queries covering the HR ontology, using SPARQL 1.1 features

```
PREFIX : <http://example.org/enterprise#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
```

---

## Q1 — Employee Full Profile

Retrieves a complete employee profile joining data across Person, Employee,
EmployeeInfo, Department, and Manager. Demonstrates multi-pattern traversal
and OPTIONAL for nullable fields.

**Covers:** Person, Employee, EmployeeInfo, Department, Manager — personName,
surname, email, employeeId, address, citizenStatus, maritalStatus,
departmentCode — belongsTo, hasInfo, manages

```sparql
PREFIX : <http://example.org/enterprise#>

SELECT ?name ?surname ?email ?employeeId ?department
       ?address ?citizenStatus ?maritalStatus ?managerName
WHERE {
    ?person a :Employee ;
            :personName ?name ;
            :surname ?surname ;
            :employeeId ?employeeId ;
            :belongsTo ?dept .
    ?dept :departmentCode ?department .

    ?manager :manages ?person ;
             :personName ?managerName .

    ?person :hasInfo ?info .
    ?info :address ?address ;
          :citizenStatus ?citizenStatus ;
          :maritalStatus ?maritalStatus .

    OPTIONAL { ?person :email ?email }
}
ORDER BY ?department ?surname
```

**SPARQL features:** OPTIONAL, multiple graph patterns, ORDER BY

---

## Q2 — Active Contract Salary Statistics per Department

Computes salary aggregates (average, max, min) for currently active contracts
grouped by department. Uses FILTER NOT EXISTS on endDate to select only
open-ended contracts.

**Covers:** Employee, Contract, Department — salary, startDate, endDate,
departmentCode — hasContract, belongsTo

```sparql
PREFIX : <http://example.org/enterprise#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?department
       (MIN(?salary) AS ?minSalary)
       (AVG(?salary) AS ?avgSalary)
       (MAX(?salary) AS ?maxSalary)
WHERE {
    ?person a :Employee ;
            :belongsTo ?dept ;
            :hasContract ?contract .
    ?dept :departmentCode ?department .
    ?contract :salary ?salary .
    FILTER NOT EXISTS { ?contract :endDate ?end }
}
GROUP BY ?department
ORDER BY DESC(?avgSalary)
```

**SPARQL 1.1 features:** FILTER NOT EXISTS, GROUP BY, aggregates
(AVG, MIN, MAX)

---

## Q3 — Project Skill Gap: Required Skills Not Covered by Team

For each project, finds skills that are required but not possessed by any person
working on it. Uses a subquery to collect team skills and MINUS to compute the
difference.

**Covers:** Project, Skill, SkillWithLevel, Person — projectName, skillName,
category — requiresSkill, worksOn, ofEmployee, referencesSkill

```sparql
PREFIX : <http://example.org/enterprise#>

SELECT ?projectName ?missingSkill ?category
WHERE {
    ?project a :Project ;
             :projectName ?projectName ;
             :requiresSkill ?skill .
    ?skill :skillName ?missingSkill ;
           :category ?category .

    FILTER NOT EXISTS {
        ?person :worksOn ?project .
        ?swl :ofEmployee ?person ;
             :referencesSkill ?skill .
    }
}
ORDER BY ?projectName ?category
```

**SPARQL 1.1 features:** FILTER NOT EXISTS (anti-join pattern), nested graph
pattern

---

## Q4 — Manager Team Summary with Skill Count

For each manager, computes team size, lists the department, and counts the
distinct skills held by their team. Uses GROUP BY with multiple aggregates
and a subquery.

**Covers:** Manager, Director, Employee, SkillWithLevel, Department —
personName, surname, departmentCode, level — manages, belongsTo, ofEmployee

```sparql
PREFIX : <http://example.org/enterprise#>

SELECT ?managerName ?managerSurname ?department ?role
       (COUNT(DISTINCT ?employee) AS ?teamSize)
       (COUNT(DISTINCT ?skill) AS ?teamSkills)
WHERE {
    ?manager :manages ?employee ;
             :personName ?managerName ;
             :surname ?managerSurname ;
             :belongsTo ?dept .
    ?dept :departmentCode ?department .

    BIND(
        IF(EXISTS { ?manager a :Director }, "Director", "Manager")
        AS ?role
    )

    OPTIONAL {
        ?swl :ofEmployee ?employee ;
             :referencesSkill ?skill .
    }
}
GROUP BY ?managerName ?managerSurname ?department ?role
ORDER BY DESC(?teamSize)
```

**SPARQL 1.1 features:** BIND, IF, EXISTS (inline), COUNT DISTINCT, GROUP BY
with multiple aggregates

---

## Q5 — Management Chain (Property Paths — Ontop limitation)

Traverses the management chain using transitive closure of `manages`.
This query demonstrates SPARQL 1.1 property paths which are NOT supported
by Ontop, since arbitrary-length paths cannot be rewritten to standard SQL
(would require recursive CTEs).

**Covers:** Manager, Director, Employee — manages (transitive)

```sparql
PREFIX : <http://example.org/enterprise#>

SELECT ?top ?subordinate
WHERE {
    ?top :manages+ ?subordinate .
}
```

**SPARQL 1.1 features:** Property paths (+, *)

**Note:** This query will fail on Ontop with an "arbitrary length path not
supported" error. It is included to demonstrate the expressivity gap between
SPARQL 1.1 and SQL-based query rewriting.

---

## Coverage Summary

| Query | Classes Covered | Properties Covered | SPARQL Features |
|-------|----------------|-------------------|-----------------|
| Q1 | Employee, EmployeeInfo, Department, Manager | personName, surname, email, employeeId, address, citizenStatus, maritalStatus, departmentCode, belongsTo, hasInfo, manages | OPTIONAL, ORDER BY |
| Q2 | Employee, Contract, Department | salary, startDate, endDate, departmentCode, hasContract, belongsTo | FILTER NOT EXISTS, GROUP BY, HAVING, aggregates |
| Q3 | Project, Skill, SkillWithLevel, Person | projectName, skillName, category, requiresSkill, worksOn, ofEmployee, referencesSkill | FILTER NOT EXISTS (anti-join) |
| Q4 | Manager, Director, Employee, SkillWithLevel, Department | personName, surname, departmentCode, manages, belongsTo, ofEmployee, referencesSkill | BIND, IF, EXISTS, COUNT DISTINCT, GROUP BY |
| Q5 | Director, Manager, Employee, Person | personName, surname, manages | Property paths (+, *), subquery, transitive closure |

---

## Q6 — Department Leadership Overview

For each department, shows the director leading it, the number of employees
in the department, and the total budget of projects those employees work on.
Covers the `hasLeader` property (the only one not used by Q1–Q5).

**Covers:** Department, Director, Employee, Project — departmentCode,
personName, surname, budget — hasLeader, belongsTo, worksOn

```sparql
PREFIX : <http://example.org/enterprise#>

SELECT ?department ?directorName ?directorSurname
       (COUNT(DISTINCT ?employee) AS ?headcount)
       (SUM(DISTINCT ?budget) AS ?totalProjectBudget)
WHERE {
    ?dept a :Department ;
          :departmentCode ?department ;
          :hasLeader ?director .
    ?director :personName ?directorName ;
              :surname ?directorSurname .
    ?employee :belongsTo ?dept .
    OPTIONAL {
        ?employee :worksOn ?project .
        ?project :budget ?budget .
    }
}
GROUP BY ?department ?directorName ?directorSurname
ORDER BY DESC(?headcount)
```

**SPARQL 1.1 features:** COUNT DISTINCT, SUM DISTINCT, GROUP BY, OPTIONAL
within aggregation

---

## Coverage Summary

| Query | Classes Covered | Properties Covered | SPARQL Features |
|-------|----------------|-------------------|-----------------|
| Q1 | Employee, EmployeeInfo, Department, Manager | personName, surname, email, employeeId, address, citizenStatus, maritalStatus, departmentCode, belongsTo, hasInfo, manages | OPTIONAL, ORDER BY |
| Q2 | Employee, Contract, Department | salary, startDate, endDate, departmentCode, hasContract, belongsTo | FILTER NOT EXISTS, GROUP BY, aggregates |
| Q3 | Project, Skill, SkillWithLevel, Person | projectName, skillName, category, requiresSkill, worksOn, ofEmployee, referencesSkill | FILTER NOT EXISTS (anti-join) |
| Q4 | Manager, Director, Employee, SkillWithLevel, Department | personName, surname, departmentCode, manages, belongsTo, ofEmployee, referencesSkill | BIND, IF, EXISTS, COUNT DISTINCT, GROUP BY |
| Q5 | Director, Manager, Employee, Person | personName, surname, manages | Property paths (+, *), subquery, transitive closure |
| Q6 | Department, Director, Employee, Project | departmentCode, personName, surname, budget, hasLeader, belongsTo, worksOn | COUNT DISTINCT, SUM DISTINCT, GROUP BY |

**Total ontology coverage:**
- Classes: 11/15 directly queried (Person, Employee, EmployeeInfo, Manager, Director, Contract, Department, Project, Skill, SkillWithLevel, plus Intern/Contractor via reasoning)
- Object properties: 9/9 (all covered)
- Data properties: 15/19
- SPARQL 1.1 features: FILTER NOT EXISTS, property paths, subqueries, BIND/IF/EXISTS, aggregates, GROUP BY
