-- HR VKG Sample Data — Standard SQL
-- Insert order respects FK dependencies

-- ============================================================
-- TPerson (20 rows: 10 employees, 5 contractors, 5 interns)
-- ============================================================
INSERT INTO TPerson (id, personName, surname, dayOfBirth, phone, email) VALUES
(1,  'Alice',    'Rossi',     '1978-03-15', '+39-02-1234567', 'alice.rossi@company.com'),
(2,  'Bob',      'Bianchi',   '1980-07-22', '+39-02-2345678', 'bob.bianchi@company.com'),
(3,  'Clara',    'Verdi',     '1985-11-03', '+39-02-3456789', 'clara.verdi@company.com'),
(4,  'David',    'Neri',      '1982-01-28', '+39-02-4567890', 'david.neri@company.com'),
(5,  'Elena',    'Russo',     '1990-06-14', '+39-02-5678901', 'elena.russo@company.com'),
(6,  'Frank',    'Colombo',   '1992-09-07', '+39-02-6789012', 'frank.colombo@company.com'),
(7,  'Giulia',   'Ferrari',   '1988-12-19', '+39-02-7890123', 'giulia.ferrari@company.com'),
(8,  'Hans',     'Muller',    '1991-04-25', '+39-02-8901234', 'hans.muller@company.com'),
(9,  'Irene',    'Esposito',  '1993-08-11', '+39-02-9012345', 'irene.esposito@company.com'),
(10, 'James',    'Romano',    '1987-02-17', '+39-02-0123456', 'james.romano@company.com'),
(11, 'Karen',    'Marino',    '1984-05-30', '+39-02-1112233', 'karen.marino@external.com'),
(12, 'Luca',     'Greco',     '1979-10-08', '+39-02-2223344', 'luca.greco@external.com'),
(13, 'Maria',    'Bruno',     '1986-03-21', '+39-02-3334455', 'maria.bruno@external.com'),
(14, 'Nicola',   'Gallo',     '1981-07-14', '+39-02-4445566', 'nicola.gallo@external.com'),
(15, 'Olivia',   'Conti',     '1983-11-26', '+39-02-5556677', 'olivia.conti@external.com'),
(16, 'Paolo',    'De Luca',   '2000-01-09', '+39-02-6667788', 'paolo.deluca@university.it'),
(17, 'Rachele',  'Mancini',   '2001-04-18', '+39-02-7778899', 'rachele.mancini@university.it'),
(18, 'Stefano',  'Ricci',     '1999-08-27', '+39-02-8889900', 'stefano.ricci@university.it'),
(19, 'Teresa',   'Moretti',   '2000-06-05', '+39-02-9990011', 'teresa.moretti@university.it'),
(20, 'Umberto',  'Barbieri',  '2001-12-13', '+39-02-1010101', 'umberto.barbieri@university.it');

-- ============================================================
-- TSkill (8 rows)
-- ============================================================
INSERT INTO TSkill (id, skillName, category) VALUES
(1, 'Java',           'Programming'),
(2, 'Python',         'Programming'),
(3, 'SQL',            'Database'),
(4, 'Project Management', 'Management'),
(5, 'Data Analysis',  'Analytics'),
(6, 'UX Design',      'Design'),
(7, 'Machine Learning','Analytics'),
(8, 'Cloud Architecture','Infrastructure');

-- ============================================================
-- TProject (6 rows: 2 HR, 2 business, 2 software)
-- ============================================================
INSERT INTO TProject (id, projectName, budget, projectType) VALUES
(1, 'Talent Acquisition Platform', 150000.00, 'hr'),
(2, 'Employee Wellness Program',    80000.00, 'hr'),
(3, 'Market Expansion APAC',       500000.00, 'business'),
(4, 'Supply Chain Optimization',   320000.00, 'business'),
(5, 'Core Banking Rewrite',        900000.00, 'software'),
(6, 'Mobile App v3',               250000.00, 'software');

-- ============================================================
-- TDepartment (5 rows)
-- director_id references TEmployee(person_id), inserted before employees
-- We temporarily set director_id = NULL, then update after TEmployee insert
-- ============================================================
INSERT INTO TDepartment (dept_code, director_id) VALUES
('HR',   NULL),
('ENG',  NULL),
('FIN',  NULL),
('MKT',  NULL),
('OPS',  NULL);

-- ============================================================
-- TEmployee (10 rows)
-- Persons 1-2: directors, Persons 3-5: managers, Persons 6-10: employees
-- manager_id references TEmployee(person_id)
-- Insert directors/managers first (self-referencing handled by NULLs initially)
-- ============================================================
INSERT INTO TEmployee (employeeId, person_id, manager_id, dept_code, role, info_id, address, citizenStatus, maritalStatus) VALUES
(1001, 1,  NULL, 'HR',  'director', 5001, '10 Via Roma, Milan',       'citizen',  'married'),
(1002, 2,  NULL, 'FIN', 'director', 5002, '25 Corso Buenos Aires, Milan', 'citizen', 'married'),
(1003, 3,  NULL, 'HR',  'manager',  5003, '8 Via Torino, Milan',      'citizen',  'single'),
(1004, 4,  NULL, 'ENG', 'manager',  5004, '42 Viale Monza, Milan',    'citizen',  'married'),
(1005, 5,  NULL, 'MKT', 'manager',  5005, '15 Via Dante, Milan',      'resident', 'single'),
(1006, 6,  NULL, 'HR',  'employee', 5006, '3 Via Manzoni, Milan',     'citizen',  'single'),
(1007, 7,  NULL, 'ENG', 'employee', 5007, '77 Corso Venezia, Milan',  'citizen',  'married'),
(1008, 8,  NULL, 'FIN', 'employee', 5008, '18 Via Garibaldi, Bergamo','resident', 'single'),
(1009, 9,  NULL, 'MKT', 'employee', 5009, '5 Via Verdi, Turin',       'citizen',  'married'),
(1010, 10, NULL, 'OPS', 'employee', 5010, '22 Via Pascoli, Rome',     'citizen',  'single');

-- Set manager_id now that all employees exist
UPDATE TEmployee SET manager_id = 2  WHERE person_id = 1;
UPDATE TEmployee SET manager_id = 1  WHERE person_id = 2;
UPDATE TEmployee SET manager_id = 1  WHERE person_id = 3;
UPDATE TEmployee SET manager_id = 2  WHERE person_id = 4;
UPDATE TEmployee SET manager_id = 1  WHERE person_id = 5;
UPDATE TEmployee SET manager_id = 3  WHERE person_id = 6;
UPDATE TEmployee SET manager_id = 4  WHERE person_id = 7;
UPDATE TEmployee SET manager_id = 2  WHERE person_id = 8;
UPDATE TEmployee SET manager_id = 5  WHERE person_id = 9;
UPDATE TEmployee SET manager_id = 4  WHERE person_id = 10;

-- Set department directors now that employees exist
UPDATE TDepartment SET director_id = 1 WHERE dept_code = 'HR';
UPDATE TDepartment SET director_id = 1 WHERE dept_code = 'ENG';
UPDATE TDepartment SET director_id = 2 WHERE dept_code = 'FIN';
UPDATE TDepartment SET director_id = 2 WHERE dept_code = 'MKT';
UPDATE TDepartment SET director_id = 1 WHERE dept_code = 'OPS';

-- ============================================================
-- TContractor (5 rows: persons 11-15)
-- ============================================================
INSERT INTO TContractor (person_id) VALUES
(11), (12), (13), (14), (15);

-- ============================================================
-- TIntern (5 rows: persons 16-20)
-- ============================================================
INSERT INTO TIntern (person_id, allowance) VALUES
(16, 800.00),
(17, 750.00),
(18, 900.00),
(19, 700.00),
(20, 850.00);

-- ============================================================
-- TContract (30 rows: each person has 1-3 contracts)
-- ============================================================
INSERT INTO TContract (person_id, contract_num, startDate, endDate, salary) VALUES
-- Employees
(1,  1, '2005-01-01', '2010-12-31', 55000.00),
(1,  2, '2011-01-01', '2018-12-31', 72000.00),
(1,  3, '2019-01-01', NULL,         95000.00),
(2,  1, '2006-03-01', '2012-02-28', 52000.00),
(2,  2, '2012-03-01', NULL,         88000.00),
(3,  1, '2010-06-01', '2016-05-31', 45000.00),
(3,  2, '2016-06-01', NULL,         62000.00),
(4,  1, '2008-09-01', '2015-08-31', 48000.00),
(4,  2, '2015-09-01', NULL,         67000.00),
(5,  1, '2015-02-01', NULL,         58000.00),
(6,  1, '2018-04-01', NULL,         42000.00),
(7,  1, '2014-07-01', '2019-06-30', 44000.00),
(7,  2, '2019-07-01', NULL,         53000.00),
(8,  1, '2017-01-01', NULL,         46000.00),
(9,  1, '2019-09-01', NULL,         41000.00),
(10, 1, '2013-11-01', '2018-10-31', 39000.00),
(10, 2, '2018-11-01', NULL,         47000.00),
-- Contractors
(11, 1, '2020-01-01', '2020-12-31', 60000.00),
(11, 2, '2021-01-01', '2022-12-31', 65000.00),
(12, 1, '2019-06-01', '2021-05-31', 70000.00),
(12, 2, '2021-06-01', NULL,         75000.00),
(13, 1, '2022-01-01', NULL,         55000.00),
(14, 1, '2020-03-01', '2021-02-28', 58000.00),
(14, 2, '2021-03-01', NULL,         62000.00),
(15, 1, '2021-09-01', NULL,         50000.00),
-- Interns
(16, 1, '2024-01-15', '2024-07-14', 9600.00),
(17, 1, '2024-02-01', '2024-07-31', 9000.00),
(18, 1, '2023-09-01', '2024-02-28', 10800.00),
(19, 1, '2024-03-01', '2024-08-31', 8400.00),
(20, 1, '2024-04-01', '2024-09-30', 10200.00);

-- ============================================================
-- TSkillWithLevel (25 rows)
-- employee_id references TEmployee(person_id)
-- ============================================================
INSERT INTO TSkillWithLevel (employee_id, skill_id, level) VALUES
(1, 4, 'expert'),
(1, 5, 'advanced'),
(1, 8, 'intermediate'),
(2, 3, 'expert'),
(2, 4, 'expert'),
(2, 5, 'advanced'),
(3, 4, 'advanced'),
(3, 6, 'intermediate'),
(4, 1, 'expert'),
(4, 2, 'advanced'),
(4, 8, 'expert'),
(5, 5, 'advanced'),
(5, 6, 'expert'),
(5, 7, 'intermediate'),
(6, 1, 'intermediate'),
(6, 3, 'beginner'),
(7, 1, 'advanced'),
(7, 2, 'expert'),
(7, 8, 'advanced'),
(8, 3, 'advanced'),
(8, 5, 'intermediate'),
(9, 6, 'advanced'),
(9, 7, 'beginner'),
(10, 2, 'intermediate'),
(10, 8, 'beginner');

-- ============================================================
-- TPersonProject (20 rows — uses project_name, not project_id)
-- ============================================================
INSERT INTO TPersonProject (person_id, project_name) VALUES
(1,  'Talent Acquisition Platform'),
(1,  'Employee Wellness Program'),
(2,  'Supply Chain Optimization'),
(3,  'Talent Acquisition Platform'),
(3,  'Employee Wellness Program'),
(4,  'Core Banking Rewrite'),
(4,  'Mobile App v3'),
(5,  'Market Expansion APAC'),
(6,  'Talent Acquisition Platform'),
(6,  'Employee Wellness Program'),
(7,  'Core Banking Rewrite'),
(7,  'Mobile App v3'),
(8,  'Supply Chain Optimization'),
(9,  'Market Expansion APAC'),
(10, 'Core Banking Rewrite'),
(11, 'Core Banking Rewrite'),
(12, 'Mobile App v3'),
(13, 'Talent Acquisition Platform'),
(14, 'Supply Chain Optimization'),
(15, 'Market Expansion APAC');

-- ============================================================
-- TProjSkill (18 rows — skills required per project)
-- References TSkillWithLevel(employee_id, skill_id)
-- ============================================================
INSERT INTO TProjSkill (project_id, employee_id, skill_id) VALUES
(1, 1, 5),
(1, 3, 6),
(1, 6, 1),
(2, 1, 4),
(2, 5, 6),
(2, 3, 4),
(3, 5, 5),
(3, 9, 6),
(3, 2, 4),
(4, 2, 3),
(4, 8, 5),
(4, 2, 5),
(5, 4, 1),
(5, 7, 1),
(5, 4, 8),
(6, 7, 2),
(6, 10, 2),
(6, 7, 8);
