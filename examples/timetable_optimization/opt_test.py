#!/usr/bin/env python
# coding: utf-8


import csv
import os
import unittest
from collections import Counter
import time
import math
import sys
from pathlib import Path

# Get inputs directory relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUTS_DIR = SCRIPT_DIR / "inputs"


def is_csv_file(filepath):
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return False  # File does not exist or is not a regular file

    # 1. Check file extension
    if not filepath.lower().endswith('.csv'):
        return False

    # 2. Attempt to sniff CSV dialect
    try:
        with open(filepath, 'r', newline='') as csvfile:
            # Read a sample of the file to help the sniffer
            sample = csvfile.read(4096)
            if not sample: # Handle empty files
                return True # An empty .csv file is still a csv file.
            
            # Reset file pointer to the beginning
            csvfile.seek(0) 
            
            # Sniff the dialect
            csv.Sniffer().sniff(sample) 
            return True
    except csv.Error:
        # If sniffing fails, it's likely not a valid CSV
        return False
    except Exception as e:
        # Catch other potential errors during file reading
        print(f"An error occurred while checking {filepath}: {e}")
        return False




def check_csv_cols(sol_dict):
    sol_cols = list(sol_dict.keys())
    if "timeslot" not in sol_cols or "class" not in sol_cols:
        print("Columns timeslot and class not found in solution.")
    
    for t in sol_dict["timeslot"]:
        if t not in timeslots:
            print("Some timeslot values are invalid!")
            print("Exiting...")
            sys.exit()
    for c in sol_dict["class"]:
        if c not in classes:
            print("Some class values are invalid!")
            print("Exiting...")
            sys.exit()

    print("Input columns all correct. Proceeding to check solution...")




def num_active_neighbours(neighbours, day_lessons):
    active_neighbours = 0
    for n in neighbours:
        if n in day_lessons:
            active_neighbours = active_neighbours + 1
    return active_neighbours




def is_consecutive(class_in, solution):
    all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == class_in]
    active_days = list(set([timeslot_days[t] for t in all_lessons]))

    for d in active_days:
        one_neighbours = 0
        zero_neighbours = 0
        day_lessons = [t for t in all_lessons if d in t]
    #     print(d, day_lessons)

        if len(day_lessons) > 1:
            for l in day_lessons:
                active_neigh = num_active_neighbours(timeslot_neighbours[l], day_lessons)
                if active_neigh == 1:
                    one_neighbours = one_neighbours + 1
                elif active_neigh == 0:
                    zero_neighbours = zero_neighbours + 1

#             print(one_neighbours, zero_neighbours)
            if zero_neighbours > 1:
#                 print("Not consecutive")
                return False
            elif one_neighbours > 2:
#                 print("Not consecutive")
                return False
    return True




def no_teacher_time_clashes(teacher, teacher_classes, solution):
    tc = teacher_classes[teacher]

    if len(tc) > 1:
        for i in range(len(tc)):
            class_i = tc[i]
            lessons_i = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == class_i]
            for j in range(len(tc)):
                if i != j:
                    class_j = tc[j]
                    lessons_j = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == class_j]
                    if not set(lessons_i).isdisjoint(lessons_j):
#                         print("Teacher time clash!")
                        return False
    return True



def is_student_time_clash(student, student_classes, solution):
    sc = student_classes[student]

    if len(sc) > 1:
        for i in range(len(sc)):
            class_i = sc[i]
            lessons_i = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == class_i]
            for j in range(len(sc)):
                if i != j:
                    class_j = sc[j]
                    lessons_j = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == class_j]
                    if not set(lessons_i).isdisjoint(lessons_j):
#                         print("Student time clash!")
                        return True
    return False



def count_student_clashes(student_list, student_classes, solution):
    num_student_clashes = 0
    for student in student_list:
        if is_student_time_clash(student, student_classes, solution):
            num_student_clashes = num_student_clashes + 1

    return num_student_clashes



def get_max_lessons_per_day(solution):
    max_lessons_per_day = {c: 0 for c in classes}
    for c in classes:
#         print(c)
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        active_days = list(set([timeslot_days[t] for t in all_lessons]))

        for d in active_days:
            day_lessons = [t for t in all_lessons if d in t]
            max_lessons_per_day[c] = max(len(day_lessons), max_lessons_per_day[c])

    return max_lessons_per_day



def expected_max_lessons_per_day(solution):
    expected_max_per_day = {c: 0 for c in classes}
    for c in classes:
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        expected_max_per_day[c] = math.ceil(len(all_lessons)/5)
    return expected_max_per_day



def get_weekdays_taught(solution):
    weekdays_taught = {c: 0 for c in classes}
    for c in classes:
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        active_days = list(set([timeslot_days[t] for t in all_lessons]))
        weekdays_taught[c] = len(active_days)
        
    return weekdays_taught



def expected_weekdays_taught(solution):
    expec_weekdays_taught = {c: 0 for c in classes}
    for c in classes:
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        expec_weekdays_taught[c] = min(5, len(all_lessons))
        
    return expec_weekdays_taught



class LessonCountTestCase(unittest.TestCase):
    def __init__(self, methodName, class_name, lesson_list, lessons_needed):
        super().__init__(methodName)
        self.class_name = class_name
        self.lesson_list = lesson_list
        self.lessons_needed = lessons_needed

    def test_lesson_counts_match(self):
        actual_count = Counter(self.lesson_list)[self.class_name]
        self.assertEqual(actual_count, self.lessons_needed[self.class_name])

def LessonCountSuite(class_name, lesson_list, lessons_needed):
    suite = unittest.TestSuite()
    suite.addTest(LessonCountTestCase('test_lesson_counts_match', class_name, lesson_list, lessons_needed))
    return suite



class ConsecutiveLessonTestCase(unittest.TestCase):
    def __init__(self, methodName, class_name, solution):
        super().__init__(methodName)
        self.class_name = class_name
        self.solution = solution

    def test_lessons_consecutive(self):
        self.assertTrue(is_consecutive(self.class_name, self.solution))

def ConsecutiveLessonSuite(class_name, solution):
    suite = unittest.TestSuite()
    suite.addTest(ConsecutiveLessonTestCase('test_lessons_consecutive', class_name, solution))
    return suite



class TeacherClashTestCase(unittest.TestCase):
    def __init__(self, methodName, teacher, teacher_classes, solution):
        super().__init__(methodName)
        self.teacher = teacher
        self.teacher_classes = teacher_classes
        self.solution = solution

    def test_no_teacher_clash(self):
        self.assertTrue(no_teacher_time_clashes(self.teacher, self.teacher_classes, self.solution))

def TeacherClashSuite(teacher, teacher_classes, solution):
    suite = unittest.TestSuite()
    suite.addTest(TeacherClashTestCase('test_no_teacher_clash', teacher, teacher_classes, solution))
    return suite



class StudentClashTestCase(unittest.TestCase):
    def __init__(self, methodName, student_list, student_classes, solution):
        super().__init__(methodName)
        self.student_list = student_list
        self.student_classes = student_classes
        self.solution = solution

    def test_num_student_clash(self):
        self.assertLessEqual(count_student_clashes(self.student_list, self.student_classes, self.solution), 10)

def StudentClashSuite(student_list, student_classes, solution):
    suite = unittest.TestSuite()
    suite.addTest(StudentClashTestCase('test_num_student_clash', student_list, student_classes, solution))
    return suite




class LessonSpreadTestCase1(unittest.TestCase):
    def __init__(self, methodName, solution):
        super().__init__(methodName)
        self.solution = solution

    def test_lesson_spread1(self):
        self.assertEqual(get_max_lessons_per_day(self.solution), expected_max_lessons_per_day(self.solution))
        
def LessonSpreadSuite1(solution):
    suite = unittest.TestSuite()
    suite.addTest(LessonSpreadTestCase1('test_lesson_spread1', solution))
    return suite




class LessonSpreadTestCase2(unittest.TestCase):
    def __init__(self, methodName, solution):
        super().__init__(methodName)
        self.solution = solution

    def test_lesson_spread2(self):
        self.assertEqual(get_weekdays_taught(self.solution), expected_weekdays_taught(self.solution))
        
def LessonSpreadSuite2(solution):
    suite = unittest.TestSuite()
    suite.addTest(LessonSpreadTestCase2('test_lesson_spread2', solution))
    return suite

def AllTestsSuite(solution, lessons_needed, student_list, student_classes, teacher_classes, teacher_list, classes):
    suite = unittest.TestSuite()
    suite.addTest(LessonSpreadTestCase1('test_lesson_spread1', solution))
    suite.addTest(LessonSpreadTestCase2('test_lesson_spread2', solution))
    suite.addTest(StudentClashTestCase('test_num_student_clash', student_list, student_classes, solution))
    for teacher in teacher_list:
        suite.addTest(TeacherClashTestCase('test_no_teacher_clash', teacher, teacher_classes, solution))
    for class_name in classes:
        suite.addTest(ConsecutiveLessonTestCase('test_lessons_consecutive', class_name, solution))
    for class_name in classes:
        suite.addTest(LessonCountTestCase('test_lesson_counts_match', class_name, solution["class"], lessons_needed))
    
    return suite


# read in data

# timeslots
with open(INPUTS_DIR / "timeslots.csv", mode ='r') as file:
    dictR = csv.DictReader(file)
    timeslots_by_day = {hdr: [] for hdr in dictR.fieldnames}
    for row in dictR:
        for field in row:
            timeslots_by_day[field].append(row[field])
timeslots = timeslots_by_day["timeslot"]
timeslot_days = {t: t[:3] for t in timeslots}

# days list
weekdays = []
with open(INPUTS_DIR / "weekdays.csv", mode ='r') as file:
    csvFile = csv.DictReader(file)
    for lines in csvFile:
        weekdays.append(lines["weekday"])
            
# class list
classes = []
with open(INPUTS_DIR / "class_list.csv", mode ='r') as file:
    csvFile = csv.DictReader(file)
    for lines in csvFile:
        classes.append(lines["class"])

# student list
student_list = []
with open(INPUTS_DIR / "student_list.csv", mode ='r') as file:
    csvFile = csv.DictReader(file)
    for lines in csvFile:
        student_list.append(lines["student"])

# student classes
student_classes = {s: [] for s in student_list}
with open(INPUTS_DIR / "student_classes.csv", mode ='r') as file:
    dictR = csv.DictReader(file)
    for row in dictR:
        student_classes[row["student"]].append(row["class"])

# teacher list
teacher_list = []
with open(INPUTS_DIR / "teacher_list.csv", mode ='r') as file:
    csvFile = csv.DictReader(file)
    for lines in csvFile:
        teacher_list.append(lines["teacher"])

# teacher classes
teacher_classes = {s: [] for s in teacher_list}
with open(INPUTS_DIR / "teacher_classes.csv", mode ='r') as file:
    dictR = csv.DictReader(file)
    for row in dictR:
        teacher_classes[row["teacher"]].append(row["class"])

# lessons required
lessons_needed = {}
with open(INPUTS_DIR / "lessons_required.csv", mode ='r') as file:
    dictR = csv.DictReader(file)
    for row in dictR:
        lessons_needed[row["class"]] = int(row["num_lessons"])



timeslot_neighbours = {}
for i in range(len(timeslots)):
    t = timeslots[i]
#     print(i, t)
    timeslot_neighbours[t] = []
    curr_day = timeslot_days[t]
    
    prev_t = timeslots[i-1]
    if timeslot_days[prev_t] == curr_day:
        timeslot_neighbours[t].append(prev_t)
    
    if i < len(timeslots)-1:
        next_t = timeslots[i+1]
        if timeslot_days[next_t] == curr_day:
            timeslot_neighbours[t].append(next_t)




sol_filename = sys.argv[1]

if not is_csv_file(sol_filename):
    print("File does not exist or is not a valid csv file.")
    print("Exiting...")
    sys.exit()
else:
    print("Valid csv found.")

with open(sol_filename, 'r', newline='') as file:
    dictR = csv.DictReader(file)
    solution = {hdr: [] for hdr in dictR.fieldnames}
    for row in dictR:
        for field in row:
            solution[field].append(row[field])

check_csv_cols(solution)


runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(AllTestsSuite(solution, lessons_needed, student_list, student_classes, teacher_classes, teacher_list, classes))

print("\n--- Test Summary ---")
print(f"Total:    {result.testsRun}")
print(f"Passed:   {result.testsRun - len(result.failures) - len(result.errors)}")
print(f"Failures: {len(result.failures)}")
print(f"Errors:   {len(result.errors)}")
for i,f in enumerate(result.failures):
    print(f"Failure {i}: {f}")
for i,e in enumerate(result.errors):
    print(f"Error {i}: {e}")
