-- HR VKG Schema — Standard SQL
-- 11 tables implementing 10 mapping patterns

-- Break circular FK (TDepartment -> TEmployee) before dropping tables
ALTER TABLE TDepartment DROP CONSTRAINT IF EXISTS fk_department_director;

DROP TABLE IF EXISTS TProjSkill;
DROP TABLE IF EXISTS TPersonProject;
DROP TABLE IF EXISTS TSkillWithLevel;
DROP TABLE IF EXISTS TContract;
DROP TABLE IF EXISTS TIntern;
DROP TABLE IF EXISTS TContractor;
DROP TABLE IF EXISTS TEmployee;
DROP TABLE IF EXISTS TDepartment;
DROP TABLE IF EXISTS TProject;
DROP TABLE IF EXISTS TSkill;
DROP TABLE IF EXISTS TPerson;

-- ============================================================
-- MpE: Person (root entity)
-- ============================================================
CREATE TABLE TPerson (
    id          INTEGER PRIMARY KEY,
    personName  VARCHAR(100) NOT NULL,
    surname     VARCHAR(100) NOT NULL,
    dayOfBirth  DATE,
    phone       VARCHAR(50),
    email       VARCHAR(100)
);

-- ============================================================
-- MpE: Skill
-- ============================================================
CREATE TABLE TSkill (
    id          INTEGER PRIMARY KEY,
    skillName   VARCHAR(100) NOT NULL,
    category    VARCHAR(50) NOT NULL
);

-- ============================================================
-- MpE + MpCE2C: Project (with discriminator for subclasses)
-- ============================================================
CREATE TABLE TProject (
    id          INTEGER PRIMARY KEY,
    projectName VARCHAR(100) NOT NULL,
    budget      DECIMAL(12,2),
    projectType VARCHAR(20) NOT NULL,
    CONSTRAINT uq_project_name UNIQUE (projectName),
    CONSTRAINT chk_project_type CHECK (projectType IN ('hr', 'business', 'software'))
);

-- ============================================================
-- MpE: Department (natural key)
-- Created without director_id FK first due to circular dependency
-- ============================================================
CREATE TABLE TDepartment (
    dept_code   VARCHAR(10) PRIMARY KEY,
    director_id INTEGER
);

-- ============================================================
-- MpHa + MpR11m + MpCE2C: Employee
-- PK = employeeId (own key); person_id is UNIQUE FK to Person
-- ============================================================
CREATE TABLE TEmployee (
    employeeId    INTEGER PRIMARY KEY,
    person_id     INTEGER NOT NULL,
    manager_id    INTEGER,
    dept_code     VARCHAR(10) NOT NULL,
    role          VARCHAR(20) NOT NULL DEFAULT 'employee',
    info_id       INTEGER NOT NULL,
    address       VARCHAR(200),
    citizenStatus VARCHAR(50),
    maritalStatus VARCHAR(50),
    CONSTRAINT uq_employee_person UNIQUE (person_id),
    CONSTRAINT uq_employee_info UNIQUE (info_id),
    CONSTRAINT chk_role CHECK (role IN ('employee', 'manager', 'director')),
    CONSTRAINT fk_employee_person FOREIGN KEY (person_id) REFERENCES TPerson(id),
    CONSTRAINT fk_employee_manager FOREIGN KEY (manager_id) REFERENCES TEmployee(person_id),
    CONSTRAINT fk_employee_dept FOREIGN KEY (dept_code) REFERENCES TDepartment(dept_code)
);

-- Add deferred FK from TDepartment to TEmployee (circular dependency)
ALTER TABLE TDepartment
    ADD CONSTRAINT fk_department_director
    FOREIGN KEY (director_id) REFERENCES TEmployee(person_id);

-- ============================================================
-- MpH: Contractor (PK = FK to Person)
-- ============================================================
CREATE TABLE TContractor (
    person_id   INTEGER PRIMARY KEY,
    CONSTRAINT fk_contractor_person FOREIGN KEY (person_id) REFERENCES TPerson(id)
);

-- ============================================================
-- MpH: Intern (PK = FK to Person)
-- ============================================================
CREATE TABLE TIntern (
    person_id   INTEGER PRIMARY KEY,
    allowance   DECIMAL(10,2),
    CONSTRAINT fk_intern_person FOREIGN KEY (person_id) REFERENCES TPerson(id)
);

-- ============================================================
-- MpEw: Contract (weak entity of Person)
-- PK includes FK to Person
-- ============================================================
CREATE TABLE TContract (
    person_id    INTEGER NOT NULL,
    contract_num INTEGER NOT NULL,
    startDate    DATE NOT NULL,
    endDate      DATE,
    salary       DECIMAL(10,2),
    CONSTRAINT pk_contract PRIMARY KEY (person_id, contract_num),
    CONSTRAINT fk_contract_person FOREIGN KEY (person_id) REFERENCES TPerson(id)
);

-- ============================================================
-- MpRR: SkillWithLevel (reified Employee–Skill)
-- FK employee_id references TEmployee(person_id) which is UNIQUE
-- ============================================================
CREATE TABLE TSkillWithLevel (
    employee_id INTEGER NOT NULL,
    skill_id    INTEGER NOT NULL,
    level       VARCHAR(20) NOT NULL,
    CONSTRAINT pk_skill_level PRIMARY KEY (employee_id, skill_id),
    CONSTRAINT fk_swl_employee FOREIGN KEY (employee_id) REFERENCES TEmployee(person_id),
    CONSTRAINT fk_swl_skill FOREIGN KEY (skill_id) REFERENCES TSkill(id)
);

-- ============================================================
-- MpRa: PersonProject (FK references projectName, not PK)
-- ============================================================
CREATE TABLE TPersonProject (
    person_id    INTEGER NOT NULL,
    project_name VARCHAR(100) NOT NULL,
    CONSTRAINT pk_person_project PRIMARY KEY (person_id, project_name),
    CONSTRAINT fk_pp_person FOREIGN KEY (person_id) REFERENCES TPerson(id),
    CONSTRAINT fk_pp_project FOREIGN KEY (project_name) REFERENCES TProject(projectName)
);

-- ============================================================
-- MpR: ProjSkill (junction for requiresSkill: Project–Skill)
-- ============================================================
CREATE TABLE TProjSkill (
    project_id  INTEGER NOT NULL,
    skill_id    INTEGER NOT NULL,
    CONSTRAINT pk_proj_skill PRIMARY KEY (project_id, skill_id),
    CONSTRAINT fk_ps_project FOREIGN KEY (project_id) REFERENCES TProject(id),
    CONSTRAINT fk_ps_skill FOREIGN KEY (skill_id) REFERENCES TSkill(id)
);
