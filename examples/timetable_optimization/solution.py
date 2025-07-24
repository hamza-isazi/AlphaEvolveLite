#!/usr/bin/env python3
"""
Timetable optimization problem - Initial solution
This program generates a basic timetable that satisfies the constraints.
"""

import csv
import os
from collections import defaultdict

def generate_timetable(inputs_dir):
    """
    Generate a basic timetable that satisfies the constraints.
    Returns a list of tuples (timeslot, class) representing the timetable.
    """
    # Load input data
    timeslots = []
    with open(os.path.join(inputs_dir, "timeslots.csv"), 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timeslots.append(row["timeslot"])
    
    classes = []
    with open(os.path.join(inputs_dir, "class_list.csv"), 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            classes.append(row["class"])
    
    lessons_needed = {}
    with open(os.path.join(inputs_dir, "lessons_required.csv"), 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            lessons_needed[row["class"]] = int(row["num_lessons"])
    
    # Simple greedy assignment: assign classes to timeslots in order
    timetable = []
    timeslot_index = 0
    
    for class_name in classes:
        lessons_required = lessons_needed[class_name]
        for _ in range(lessons_required):
            if timeslot_index < len(timeslots):
                timetable.append((timeslots[timeslot_index], class_name))
                timeslot_index += 1
            else:
                # If we run out of timeslots, break
                break
    
    return timetable

def save_timetable(timetable, output_file="solution.csv"):
    """Save the timetable to a CSV file."""
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timeslot", "class"])
        for timeslot, class_name in timetable:
            writer.writerow([timeslot, class_name])

def main(inputs_dir="inputs"):
    """Main function to generate and save the timetable."""
    timetable = generate_timetable(inputs_dir)
    save_timetable(timetable)
    print(f"Generated timetable with {len(timetable)} assignments")
    return timetable

if __name__ == "__main__":
    main() 