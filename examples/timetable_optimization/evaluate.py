#!/usr/bin/env python3
"""
Evaluation function for the timetable optimization problem.
This function takes a program path and returns a score based on how well
the generated timetable satisfies the optimization criteria.
"""

import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from collections import Counter
import math
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUTS_DIR = SCRIPT_DIR / "inputs"

# Global variables for test data
timeslots = []
timeslot_days = {}
timeslot_neighbours = {}
weekdays = []
classes = []
student_list = []
student_classes = {}
teacher_list = []
teacher_classes = {}
lessons_needed = {}

def load_test_data():
    """Load all the test data from CSV files."""
    global timeslots, timeslot_days, timeslot_neighbours, weekdays, classes
    global student_list, student_classes, teacher_list, teacher_classes, lessons_needed
    
    # Load timeslots
    with open(INPUTS_DIR / "timeslots.csv", mode='r') as file:
        dictR = csv.DictReader(file)
        timeslots_by_day = {hdr: [] for hdr in dictR.fieldnames}
        for row in dictR:
            for field in row:
                timeslots_by_day[field].append(row[field])
    timeslots = timeslots_by_day["timeslot"]
    timeslot_days = {t: t[:3] for t in timeslots}
    
    # Load weekdays
    with open(INPUTS_DIR / "weekdays.csv", mode='r') as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            weekdays.append(lines["weekday"])
    
    # Load classes
    with open(INPUTS_DIR / "class_list.csv", mode='r') as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            classes.append(lines["class"])
    
    # Load students
    with open(INPUTS_DIR / "student_list.csv", mode='r') as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            student_list.append(lines["student"])
    
    # Load student classes
    student_classes = {s: [] for s in student_list}
    with open(INPUTS_DIR / "student_classes.csv", mode='r') as file:
        dictR = csv.DictReader(file)
        for row in dictR:
            student_classes[row["student"]].append(row["class"])
    
    # Load teachers
    with open(INPUTS_DIR / "teacher_list.csv", mode='r') as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            teacher_list.append(lines["teacher"])
    
    # Load teacher classes
    teacher_classes = {s: [] for s in teacher_list}
    with open(INPUTS_DIR / "teacher_classes.csv", mode='r') as file:
        dictR = csv.DictReader(file)
        for row in dictR:
            teacher_classes[row["teacher"]].append(row["class"])
    
    # Load lessons required
    with open(INPUTS_DIR / "lessons_required.csv", mode='r') as file:
        dictR = csv.DictReader(file)
        for row in dictR:
            lessons_needed[row["class"]] = int(row["num_lessons"])
    
    # Build timeslot neighbours
    timeslot_neighbours = {}
    for i in range(len(timeslots)):
        t = timeslots[i]
        timeslot_neighbours[t] = []
        curr_day = timeslot_days[t]
        
        prev_t = timeslots[i-1]
        if timeslot_days[prev_t] == curr_day:
            timeslot_neighbours[t].append(prev_t)
        
        if i < len(timeslots)-1:
            next_t = timeslots[i+1]
            if timeslot_days[next_t] == curr_day:
                timeslot_neighbours[t].append(next_t)

def is_csv_file(filepath):
    """Check if a file is a valid CSV file."""
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return False

    if not filepath.lower().endswith('.csv'):
        return False

    try:
        with open(filepath, 'r', newline='') as csvfile:
            sample = csvfile.read(4096)
            if not sample:
                return True
            
            csvfile.seek(0)
            csv.Sniffer().sniff(sample)
            return True
    except (csv.Error, Exception):
        return False

def check_csv_cols(solution):
    """Check that the solution CSV has the required columns and valid data."""
    sol_cols = list(solution.keys())
    if "timeslot" not in sol_cols or "class" not in sol_cols:
        print("Columns timeslot and class not found in solution.")
        return False
    
    for t in solution["timeslot"]:
        if t not in timeslots:
            print("Some timeslot values are invalid!")
            return False
    for c in solution["class"]:
        if c not in classes:
            print("Some class values are invalid!")
            return False
    
    print("Input columns all correct. Proceeding to check solution...")
    return True

def num_active_neighbours(neighbours, day_lessons):
    """Count active neighbouring timeslots."""
    active_neighbours = 0
    for n in neighbours:
        if n in day_lessons:
            active_neighbours = active_neighbours + 1
    return active_neighbours

def is_consecutive(class_in, solution):
    """Check if lessons for a class are consecutive within each day."""
    all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == class_in]
    active_days = list(set([timeslot_days[t] for t in all_lessons]))

    for d in active_days:
        one_neighbours = 0
        zero_neighbours = 0
        day_lessons = [t for t in all_lessons if d in t]

        if len(day_lessons) > 1:
            for l in day_lessons:
                active_neigh = num_active_neighbours(timeslot_neighbours[l], day_lessons)
                if active_neigh == 1:
                    one_neighbours = one_neighbours + 1
                elif active_neigh == 0:
                    zero_neighbours = zero_neighbours + 1

            if zero_neighbours > 1:
                return False
            elif one_neighbours > 2:
                return False
    return True

def no_teacher_time_clashes(teacher, teacher_classes, solution):
    """Check that a teacher has no time clashes."""
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
                        return False
    return True

def is_student_time_clash(student, student_classes, solution):
    """Check if a student has time clashes."""
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
                        return True
    return False

def count_student_clashes(student_list, student_classes, solution):
    """Count the number of students with time clashes."""
    num_student_clashes = 0
    for student in student_list:
        if is_student_time_clash(student, student_classes, solution):
            num_student_clashes = num_student_clashes + 1
    return num_student_clashes

def get_max_lessons_per_day(solution):
    """Get the maximum number of lessons per day for each class."""
    max_lessons_per_day = {c: 0 for c in classes}
    for c in classes:
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        active_days = list(set([timeslot_days[t] for t in all_lessons]))

        for d in active_days:
            day_lessons = [t for t in all_lessons if d in t]
            max_lessons_per_day[c] = max(len(day_lessons), max_lessons_per_day[c])
    return max_lessons_per_day

def expected_max_lessons_per_day(solution):
    """Calculate expected maximum lessons per day for each class."""
    expected_max_per_day = {c: 0 for c in classes}
    for c in classes:
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        expected_max_per_day[c] = math.ceil(len(all_lessons)/5)
    return expected_max_per_day

def get_weekdays_taught(solution):
    """Get the number of weekdays each class is taught."""
    weekdays_taught = {c: 0 for c in classes}
    for c in classes:
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        active_days = list(set([timeslot_days[t] for t in all_lessons]))
        weekdays_taught[c] = len(active_days)
    return weekdays_taught

def expected_weekdays_taught(solution):
    """Calculate expected number of weekdays each class should be taught."""
    expec_weekdays_taught = {c: 0 for c in classes}
    for c in classes:
        all_lessons = [timeslot for timeslot, cls in zip(solution['timeslot'], solution['class']) if cls == c]
        expec_weekdays_taught[c] = min(5, len(all_lessons))
    return expec_weekdays_taught

# Test case classes
class LessonCountTestCase(unittest.TestCase):
    def __init__(self, methodName, class_name, lesson_list, lessons_needed):
        super().__init__(methodName)
        self.class_name = class_name
        self.lesson_list = lesson_list
        self.lessons_needed = lessons_needed

    def test_lesson_counts_match(self):
        actual_count = Counter(self.lesson_list)[self.class_name]
        self.assertEqual(actual_count, self.lessons_needed[self.class_name])

class ConsecutiveLessonTestCase(unittest.TestCase):
    def __init__(self, methodName, class_name, solution):
        super().__init__(methodName)
        self.class_name = class_name
        self.solution = solution

    def test_lessons_consecutive(self):
        self.assertTrue(is_consecutive(self.class_name, self.solution))

class TeacherClashTestCase(unittest.TestCase):
    def __init__(self, methodName, teacher, teacher_classes, solution):
        super().__init__(methodName)
        self.teacher = teacher
        self.teacher_classes = teacher_classes
        self.solution = solution

    def test_no_teacher_clash(self):
        self.assertTrue(no_teacher_time_clashes(self.teacher, self.teacher_classes, self.solution))

class StudentClashTestCase(unittest.TestCase):
    def __init__(self, methodName, student_list, student_classes, solution):
        super().__init__(methodName)
        self.student_list = student_list
        self.student_classes = student_classes
        self.solution = solution

    def test_num_student_clash(self):
        self.assertLessEqual(count_student_clashes(self.student_list, self.student_classes, self.solution), 10)

class LessonSpreadTestCase1(unittest.TestCase):
    def __init__(self, methodName, solution):
        super().__init__(methodName)
        self.solution = solution

    def test_lesson_spread1(self):
        self.assertEqual(get_max_lessons_per_day(self.solution), expected_max_lessons_per_day(self.solution))

class LessonSpreadTestCase2(unittest.TestCase):
    def __init__(self, methodName, solution):
        super().__init__(methodName)
        self.solution = solution

    def test_lesson_spread2(self):
        self.assertEqual(get_weekdays_taught(self.solution), expected_weekdays_taught(self.solution))

def create_test_suite(solution):
    """Create a test suite with all the timetable tests."""
    suite = unittest.TestSuite()
    
    # Add all test cases
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

def _load_module(path: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate(path: str) -> float:
    """
    Evaluate a timetable generation program.
    
    Args:
        path: Path to the Python program that generates a timetable
        
    Returns:
        float: Score between 0.0 and 1.0, where 1.0 is perfect
    """
    # Load test data
    load_test_data()
    
    # Load the candidate module
    mod = _load_module(path)
    
    # Create a temporary directory for the solution
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to a temp directory so we can clean up after ourselves and run the program
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:            
            # Run the main function from the module
            if hasattr(mod, 'main'):
                timetable = mod.main(INPUTS_DIR)
            else:
                # If no main function, try to call generate_timetable directly
                if hasattr(mod, 'generate_timetable'):
                    timetable = mod.generate_timetable(INPUTS_DIR)
                    mod.save_timetable(timetable, "solution.csv")
                else:
                    return 0.0
            
            # Check if solution.csv was created
            if not os.path.exists("solution.csv"):
                return 0.0
            
            # Load the solution
            solution_path = os.path.join(temp_dir, "solution.csv")
            
            if not is_csv_file(solution_path):
                print("File does not exist or is not a valid csv file.")
                return 0.0
            
            with open(solution_path, 'r', newline='') as file:
                dictR = csv.DictReader(file)
                solution = {hdr: [] for hdr in dictR.fieldnames}
                for row in dictR:
                    for field in row:
                        solution[field].append(row[field])
            
            # Check CSV columns
            if not check_csv_cols(solution):
                return 0.0
            
            # Run the tests
            runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
            result = runner.run(create_test_suite(solution))
            
            # Calculate score as fraction of passed tests
            total_tests = result.testsRun
            passed_tests = result.testsRun - len(result.failures) - len(result.errors)
            
            if total_tests == 0:
                return 0.0
            
            score = passed_tests / total_tests
            
            # Print test summary
            print("=== EVAL OUTPUT ===")
            print(f"Total:    {total_tests}")
            print(f"Passed:   {passed_tests}")
            print(f"Failures: {len(result.failures)}")
            print(f"Errors:   {len(result.errors)}")
            for i, f in enumerate(result.failures):
                print(f"Failure {i}: {f}")
            for i, e in enumerate(result.errors):
                print(f"Error {i}: {e}")
            print("=== END EVAL OUTPUT ===")
            
            return max(0.0, min(1.0, score))
            
        finally:
            # Make sure we return to the original directory
            os.chdir(original_cwd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    
    score = evaluate(sys.argv[1])
    print(f"Score: {score}") 