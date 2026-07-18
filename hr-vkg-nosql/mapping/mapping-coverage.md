# VKG Mapping Pattern Coverage Analysis

> HR Ontology (OWL 2 QL) vs. VKG Mapping Patterns from slides 184–256

|             | Count |
|-------------|-------|
| Classes     | 15    |
| Associations| 9     |
| Data Props  | 19    |
| **Distinct Patterns** | **10** |
| SQL Tables  | 11    |

All 15 classes, 9 object properties, and 19 data properties are fully covered
using **10 distinct VKG mapping patterns** (the spec requires at least 3).
All available patterns from the slides are covered.

---

## Patterns Used

| Pattern | Name                    | Applied to                                              |
|---------|-------------------------|---------------------------------------------------------|
| MpE     | Entity                  | Person, Project, Department, Skill (4 classes)          |
| MpH     | Hierarchy               | Contractor, Intern (2 classes)                          |
| MpHa    | Hierarchy with Id. Alignment | Employee (1 class, own PK ≠ FK to Person)          |
| MpR     | Relationship            | requiresSkill (1 assoc)                                 |
| MpRa    | Rel. with Id. Alignment | worksOn (1 assoc, FK references projectName not PK)     |
| MpRm    | Rel. with Merging       | manages, hasLeader, belongsTo (3 assocs)                |
| MpR11m  | 1-1 Rel. with Merging   | EmployeeInfo + hasInfo (1 class, from same TEmployee)   |
| MpEw    | Entity with Weak Id.    | Contract + hasContract (1 class, PK includes FK to Person) |
| MpRR    | Reified Relationship    | SkillWithLevel + ofEmployee + referencesSkill (1 class) |
| MpCE2C  | Clustering Entity to Class | Manager, Director, HRProject, BusinessProject, SoftwareProject (5 subclasses) |

---

## Class Mapping

| Class           | Pattern | Data Properties                             | SQL Design Note                                  |
|-----------------|---------|---------------------------------------------|--------------------------------------------------|
| Person          | MpE     | personName, surname, dayOfBirth, phone, email | Root entity; own table with PK                 |
| Employee        | MpHa    | employeeId                                  | Own PK (employeeId), FK person_id → Person   |
| EmployeeInfo    | MpR11m  | address, citizenStatus, maritalStatus       | Additional unique key info_id in TEmployee       |
| Manager         | MpCE2C  | (inherited from Employee)                   | WHERE role IN ('manager','director') on TEmployee|
| Director        | MpCE2C  | (inherited from Employee)                   | WHERE role = 'director' on TEmployee             |
| Contractor      | MpH     | (none)                                      | PK is FK to Person (ISA hierarchy)               |
| Intern          | MpH     | allowance                                   | PK is FK to Person (ISA hierarchy)               |
| Contract        | MpEw    | startDate, endDate, salary                  | Weak entity: PK = (person_id, contract_num)      |
| Project         | MpE     | projectName, budget                         | Own table with PK + projectType discriminator    |
| HRProject       | MpCE2C  | (inherited from Project)                    | WHERE projectType = 'hr' on TProject             |
| BusinessProject | MpCE2C  | (inherited from Project)                    | WHERE projectType = 'business' on TProject       |
| SoftwareProject | MpCE2C  | (inherited from Project)                    | WHERE projectType = 'software' on TProject       |
| Department      | MpE     | departmentCode                              | Own table; departmentCode as natural PK          |
| SkillWithLevel  | MpRR    | level                                       | Reified relationship: PK = (employee_id, skill_id)|
| Skill           | MpE     | skillName, category                         | Own table with surrogate PK                      |

---

## Association Mapping

| Property       | Domain → Range              | Cardinality | Pattern | SQL Design Note                          |
|----------------|-----------------------------|-------------|---------|------------------------------------------|
| hasContract    | Person → Contract           | 1 : 1..*    | MpEw    | FK person_id is part of Contract's PK    |
| worksOn        | Person → Project            | \* : \*     | MpRa    | Junction table TPersonProject (FK→projectName, not PK) |
| ofEmployee     | SkillWithLevel → Employee   | \* : 1      | MpRR    | Part of reified PK (employee_id)         |
| referencesSkill| SkillWithLevel → Skill      | 1..* : 1    | MpRR    | Part of reified PK (skill_id)            |
| requiresSkill  | Project → Skill             | \* : \*     | MpR     | Junction table TProjSkill                |
| manages        | Manager → Employee          | 1 : 1..*    | MpRm    | FK manager_id merged into TEmployee (self-ref) |
| hasLeader      | Department → Director       | 1..* : 1    | MpRm    | FK director_id merged into TDepartment   |
| belongsTo      | Employee → Department       | 1..* : 1    | MpRm    | FK dept_code merged into TEmployee       |
| hasInfo        | Employee → EmployeeInfo     | 1 : 1       | MpR11m  | Both keys (person_id, info_id) in TEmployee |

---

## Pattern Details

### MpE — Entity Pattern (4 classes)

Maps a single table to a class with its data properties. The primary key
constructs the IRI template.

**Applied to:** Person, Project, Department, Skill.

```
mappingId MPerson
source  SELECT id, personName, surname, dayOfBirth, phone, email
        FROM TPerson
target  :person/{id} rdf:type :Person ;
          :personName {personName} ;
          :surname {surname} ;
          :dayOfBirth {dayOfBirth} ;
          :phone {phone} ;
          :email {email} .
```

### MpH — Hierarchy Pattern (2 classes)

Maps a table whose PK is a FK to a parent table, representing an ISA (subclass)
relationship. The child inherits the parent's IRI template. PK = FK (same key).

**Applied to:** Contractor (PK→Person), Intern (PK→Person).

```
mappingId MContractor
source  SELECT person_id FROM TContractor
target  :person/{person_id} rdf:type :Contractor .

mappingId MIntern
source  SELECT person_id, allowance FROM TIntern
target  :person/{person_id} rdf:type :Intern ;
          :allowance {allowance} .
```

> **Disjointness:** Employee, Contractor, and Intern are declared disjoint in the
> ontology. This is enforced at the ontology axiom level, not in the mapping.

### MpHa — Hierarchy with Identifier Alignment Pattern (1 class)

Maps a child table that has its own independent PK, distinct from the FK to the
parent table. The mapping must use the FK (not the PK) to construct the IRI,
"aligning" the table's identity system with the ontology's identity system.

**Applied to:** Employee (own PK `employeeId`, FK `person_id` → Person).

```
mappingId MEmployee
source  SELECT person_id, employeeId FROM TEmployee
target  :person/{person_id} rdf:type :Employee ;
          :employeeId {employeeId} .
```

> **Key insight:** The table's PK is `employeeId`, but the Employee's ontological
> identity comes from `person_id` (the FK to Person), since Employee ⊑ Person.
> The mapping "aligns" identifiers by using `person_id` (not the table PK) in the
> IRI template. This is what distinguishes MpHa from MpH (where PK = FK).

### MpR — Relationship Pattern (1 association)

Maps a separate junction table (with composite FK as PK) to an object property.
Used for many-to-many relationships where both FKs reference primary keys.

**Applied to:** requiresSkill (Project–Skill).

```
mappingId MRequiresSkill
source  SELECT project_id, skill_id FROM TProjSkill
target  :project/{project_id} :requiresSkill :skill/{skill_id} .
```

### MpRa — Relationship with Identifier Alignment Pattern (1 association)

Maps a junction table where one FK references a non-primary key (candidate key)
of the target entity. Because the entity's IRI is built from its PK, the mapping
requires a JOIN to align the alternative key back to the PK.

**Applied to:** worksOn (Person–Project). The junction table references
`projectName` (UNIQUE) instead of Project's PK `id`.

```
mappingId MWorksOn
source  SELECT pp.person_id, p.id
        FROM TPersonProject pp
        JOIN TProject p ON pp.project_name = p.projectName
target  :person/{person_id} :worksOn :project/{p.id} .
```

> **Key insight:** The junction table `TPersonProject` stores `project_name`
> (referencing the UNIQUE candidate key `TProject.projectName`), not `project_id`.
> The mapping must JOIN back to `TProject` to recover the PK `id` needed for the
> IRI `:project/{id}`. This JOIN for identifier alignment is what distinguishes
> MpRa from MpR.

### MpRm — Relationship with Merging Pattern (3 associations)

Maps a FK column merged into an entity table to an object property. Used when one
side has (x, 1) cardinality (at most 1).

**Applied to:** manages (FK manager_id in TEmployee, self-referencing),
hasLeader (FK director_id in TDepartment), belongsTo (FK dept_code in TEmployee).

```
mappingId MBelongsTo
source  SELECT person_id, dept_code FROM TEmployee
target  :person/{person_id} :belongsTo :department/{dept_code} .

mappingId MManages
source  SELECT person_id, manager_id FROM TEmployee
        WHERE manager_id IS NOT NULL
target  :person/{manager_id} :manages :person/{person_id} .
```

### MpR11m — 1-1 Relationship with Merging Pattern (1 class + 1 association)

Maps two classes from a single table that has two distinct keys. Each key
identifies a different class, and their 1:1 relationship is implicit in the
shared rows. Domain knowledge determines which attributes belong to which class.

**Applied to:** EmployeeInfo (from TEmployee, identified by info_id).
Covers the `hasInfo` association.

The TEmployee table contains both:
- `employeeId` (PK) — the table's own primary key
- `person_id` (UNIQUE, FK) — identifies the Employee in the ontology (MpHa)
- `info_id` (additional unique key) — identifies the EmployeeInfo
- Employee attributes: employeeId, manager_id, dept_code, role
- EmployeeInfo attributes: address, citizenStatus, maritalStatus

```
mappingId MEmployeeInfo
source  SELECT info_id, address, citizenStatus, maritalStatus
        FROM TEmployee
target  :empinfo/{info_id} rdf:type :EmployeeInfo ;
          :address {address} ;
          :citizenStatus {citizenStatus} ;
          :maritalStatus {maritalStatus} .

mappingId MHasInfo
source  SELECT person_id, info_id FROM TEmployee
target  :person/{person_id} :hasInfo :empinfo/{info_id} .
```

> **Key insight:** Both Employee and EmployeeInfo are extracted from the same
> TEmployee table, each using a different key for its IRI template. This is what
> distinguishes MpR11m from MpE (one table, one class) or MpRm (FK in a
> different table). The ontology axioms `∃hasInfo ≡ Employee` and
> `∃hasInfo⁻ ≡ EmployeeInfo` capture the 1:1 correspondence.

### MpEw — Entity with Weak Identification Pattern (1 class + 1 association)

Maps a weak entity whose PK includes a FK to a parent entity. The entity cannot
be uniquely identified without reference to its parent. The IRI template uses
both the local key and the FK.

**Applied to:** Contract (weak entity of Person). Covers the `hasContract`
association.

The TContract table has:
- PK = (person_id, contract_num) — composite, includes the FK
- FK person_id → TPerson(id)
- Additional attributes: startDate, endDate, salary

```
mappingId MContract
source  SELECT person_id, contract_num, startDate, endDate, salary
        FROM TContract
target  :contract/{person_id}/{contract_num} rdf:type :Contract ;
          :startDate {startDate} ;
          :endDate {endDate} ;
          :salary {salary} .

mappingId MHasContract
source  SELECT person_id, contract_num FROM TContract
target  :person/{person_id} :hasContract :contract/{person_id}/{contract_num} .
```

> **Key insight:** The IRI `:contract/{person_id}/{contract_num}` includes the FK
> to Person as part of the identity. This is what distinguishes MpEw from MpE
> (independent PK) — the entity's existence depends on its parent. The mapping
> also produces the `hasContract` relationship as a byproduct.

### MpRR — Reified Relationship Pattern (1 class + 2 associations)

Maps a relationship table that has its own attributes into a class. SkillWithLevel
reifies the Employee–Skill relationship, with `level` as the relationship's own
data property. The PK is the composite of FKs to both participating entities.

**Applied to:** SkillWithLevel (reified Employee–Skill). Covers `ofEmployee` and
`referencesSkill` associations.

```
mappingId MSkillWithLevel
source  SELECT employee_id, skill_id, level
        FROM TSkillWithLevel
target  :swl/{employee_id}/{skill_id} rdf:type :SkillWithLevel ;
          :ofEmployee :person/{employee_id} ;
          :referencesSkill :skill/{skill_id} ;
          :level {level} .
```

> **Key insight:** The IRI `:swl/{employee_id}/{skill_id}` is constructed from
> both FKs, and `level` is an additional attribute of the reified class — this is
> what distinguishes MpRR from MpR.

### MpCE2C — Clustering Entity to Class Pattern (5 subclasses)

Maps subclasses derived by filtering on a discriminator column. No separate
tables — the same base table is filtered with WHERE clauses. Used twice: for
Employee roles and Project types.

**Applied to:** Manager, Director (role column in TEmployee); HRProject,
BusinessProject, SoftwareProject (projectType column in TProject).

#### Employee role clustering

```
mappingId MManager
source  SELECT person_id FROM TEmployee
        WHERE role IN ('manager', 'director')
target  :person/{person_id} rdf:type :Manager .

mappingId MDirector
source  SELECT person_id FROM TEmployee
        WHERE role = 'director'
target  :person/{person_id} rdf:type :Director .
```

> **Director IS-A Manager cascade:** Director uses `WHERE role = 'director'`,
> while Manager uses `WHERE role IN ('manager', 'director')`. This ensures every
> Director is also classified as a Manager, preserving Director ⊑ Manager ⊑ Employee.

#### Project type clustering

```
mappingId MProject
source  SELECT id, projectName, budget FROM TProject
target  :project/{id} rdf:type :Project ;
          :projectName {projectName} ;
          :budget {budget} .

mappingId MHRProject
source  SELECT id FROM TProject WHERE projectType = 'hr'
target  :project/{id} rdf:type :HRProject .

mappingId MBusinessProject
source  SELECT id FROM TProject WHERE projectType = 'business'
target  :project/{id} rdf:type :BusinessProject .

mappingId MSoftwareProject
source  SELECT id FROM TProject WHERE projectType = 'software'
target  :project/{id} rdf:type :SoftwareProject .
```

> **Key distinction from MpH:** MpCE2C uses a single table + WHERE filter (no
> join, no FK). MpH uses a separate child table whose PK is a FK to the parent.
> Both produce subclass axioms, but from different SQL structures.

---

## SQL Schema (11 tables)

```sql
TPerson         (id, personName, surname, dayOfBirth, phone, email)

TEmployee       (employeeId, person_id, manager_id, dept_code, role,
                 info_id, address, citizenStatus, maritalStatus)
                  PK employeeId                            -- own independent PK
                  UNIQUE person_id                         -- 1:1 with Person (MpHa)
                  UNIQUE info_id                           -- MpR11m
                  FK person_id  -> TPerson(id)             -- hierarchy link (not PK!)
                  FK manager_id -> TEmployee(person_id)    -- MpRm (self-ref)
                  FK dept_code  -> TDepartment(dept_code)  -- MpRm
                  role IN ('employee','manager','director') -- MpCE2C

TContractor     (person_id)
                  FK person_id  -> TPerson(id)            -- MpH

TIntern         (person_id, allowance)
                  FK person_id  -> TPerson(id)            -- MpH

TContract       (person_id, contract_num, startDate, endDate, salary)
                  PK (person_id, contract_num)            -- MpEw
                  FK person_id  -> TPerson(id)

TProject        (id, projectName, budget, projectType)
                  UNIQUE projectName                       -- candidate key for MpRa
                  projectType IN ('hr','business','software') -- MpCE2C

TDepartment     (dept_code, director_id)
                  FK director_id -> TEmployee(person_id)  -- MpRm (references UNIQUE)

TSkill          (id, skillName, category)

TSkillWithLevel (employee_id, skill_id, level)            -- MpRR
                  PK (employee_id, skill_id)
                  FK employee_id -> TEmployee(person_id)  -- references UNIQUE, not PK
                  FK skill_id   -> TSkill(id)

TPersonProject  (person_id, project_name)                 -- MpRa
                  FK person_id    -> TPerson(id)
                  FK project_name -> TProject(projectName) -- references UNIQUE, not PK

TProjSkill      (project_id, skill_id)                     -- MpR
                  FK project_id -> TProject(id)
                  FK skill_id   -> TSkill(id)
```

---

## Pattern Coverage

All 10 VKG mapping patterns from the slides are now applied:

| # | Pattern | Name                         |
|---|---------|------------------------------|
| 1 | MpE     | Entity                       |
| 2 | MpH     | Hierarchy                    |
| 3 | MpHa    | Hierarchy with Id. Alignment |
| 4 | MpR     | Relationship                 |
| 5 | MpRa    | Rel. with Id. Alignment      |
| 6 | MpRm    | Rel. with Merging            |
| 7 | MpR11m  | 1-1 Rel. with Merging        |
| 8 | MpEw    | Entity with Weak Id.         |
| 9 | MpRR    | Reified Relationship         |
|10 | MpCE2C  | Clustering Entity to Class   |
