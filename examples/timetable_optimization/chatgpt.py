import csv
import random
from collections import defaultdict

# Load the set of classes
classes_set = set()
with open('class_list.csv', newline='') as f:
    for row in csv.DictReader(f):
        classes_set.add(row['class'])

# Define weekdays in order
days_order = ['MON', 'TUE', 'WED', 'THU', 'FRI']

# Load timeslots and group by day
timeslot_days = {}
timeslot_list = []
with open('timeslots.csv', newline='') as f:
    for row in csv.DictReader(f):
        t = row['timeslot']
        timeslot_list.append(t)
        timeslot_days[t] = row['weekday']

def time_key(ts: str) -> int:
    # Convert e.g. "MON 07_45 - 08_30" to 745 for sorting
    parts = ts.split()
    start = parts[1]  # '07_45'
    h, m = start.split('_')
    return int(h) * 100 + int(m)

# Group timeslots per day and sort them by start time
slots_by_day: dict[str, list[str]] = {d: [] for d in days_order}
for ts in timeslot_list:
    d = timeslot_days[ts]
    if d in slots_by_day:
        slots_by_day[d].append(ts)
for d in days_order:
    slots_by_day[d] = sorted(slots_by_day[d], key=time_key)

# Map teachers to their classes and vice‑versa
teacher_classes: dict[str, list[str]] = {}
with open('teacher_list.csv', newline='') as f:
    for row in csv.DictReader(f):
        teacher_classes[row['teacher']] = []
with open('teacher_classes.csv', newline='') as f:
    for row in csv.DictReader(f):
        cls = row['class']
        if cls in classes_set:
            teacher_classes[row['teacher']].append(cls)
# Reverse mapping: class -> teacher
class_teacher = {cls: teacher for teacher, cls_list in teacher_classes.items()
                 for cls in cls_list}

# Map students to classes and classes to students
student_list: list[str] = []
student_classes: dict[str, list[str]] = {}
with open('student_list.csv', newline='') as f:
    for row in csv.DictReader(f):
        s = row['student']
        student_list.append(s)
        student_classes[s] = []
with open('student_classes.csv', newline='') as f:
    for row in csv.DictReader(f):
        c = row['class']
        if c in classes_set:
            student_classes[row['student']].append(c)

# Build class -> set of students
class_students = {c: set() for c in classes_set}
for student, cls_list in student_classes.items():
    for cls in cls_list:
        class_students[cls].add(student)

# Read how many lessons each class needs
lessons_needed: dict[str, int] = {}
with open('lessons_required.csv', newline='') as f:
    for row in csv.DictReader(f):
        lessons_needed[row['class']] = int(row['num_lessons'])

# Precompute conflict degree (number of other classes sharing students) for heuristics
class_conflicts = {c: set() for c in classes_set}
for student, cls_list in student_classes.items():
    cls_filtered = [c for c in cls_list if c in classes_set]
    for i in range(len(cls_filtered)):
        for j in range(i + 1, len(cls_filtered)):
            c1, c2 = cls_filtered[i], cls_filtered[j]
            class_conflicts[c1].add(c2)
            class_conflicts[c2].add(c1)
conflict_degree = {c: len(class_conflicts[c]) for c in classes_set}

def evaluate(schedule: dict[str, list[str]]) -> tuple[int, int]:
    """Returns (teacher_conflicts, number_of_students_with_clashes)."""
    from collections import defaultdict
    teacher_timeslot: defaultdict[str, list[str]] = defaultdict(list)
    for cls, times in schedule.items():
        teacher = class_teacher[cls]
        for t in times:
            teacher_timeslot[teacher].append(t)
    # teacher conflicts
    for times in teacher_timeslot.values():
        counts = defaultdict(int)
        for t in times:
            counts[t] += 1
        if any(v > 1 for v in counts.values()):
            # A teacher has two classes at the same timeslot
            return 1, None

    # student clashes (students with any timeslot containing >= 2 of their classes)
    student_slots: dict[str, dict[str, int]] = {s: defaultdict(int) for s in student_list}
    for cls, times in schedule.items():
        for t in times:
            for s in class_students[cls]:
                student_slots[s][t] += 1

    student_clashes = sum(
        1 for s in student_list
        if any(v >= 2 for v in student_slots[s].values())
    )
    return 0, student_clashes

# Main search routine: randomise day assignments then assign slots greedily + local search
best_schedule = None
best_clashes = float('inf')
random.seed(42)

for run in range(30):  # adjust the number of runs as needed
    # Randomly distribute each class's lessons across the five days
    day_sessions = {c: {d: 0 for d in days_order} for c in classes_set}
    for cls in classes_set:
        n = lessons_needed[cls]
        if n <= 5:
            days = random.sample(days_order, n)
            for d in days:
                day_sessions[cls][d] = 1
        else:
            base = n // 5
            extra = n % 5
            for d in days_order:
                day_sessions[cls][d] = base
            # assign the extra sessions to randomly selected days
            extra_days = random.sample(days_order, extra)
            for d in extra_days:
                day_sessions[cls][d] += 1

    schedule: dict[str, list[str]] = {c: [] for c in classes_set}
    success = True

    # For each day, assign classes to timeslots
    for d in days_order:
        # Tasks that need a lesson on this day
        tasks = {c: cnt for c, cnt in ((c, day_sessions[c][d]) for c in classes_set) if cnt}

        # teacher occupancy (per day) in slot indices (0-8)
        teacher_occ = defaultdict(set)
        # assignment mapping: class -> starting slot index (for length 1 or 2)
        assignments: dict[str, int] = {}

        # student slot counts per day (used to estimate collisions)
        student_slot_count = {s: defaultdict(int) for s in student_list}

        # Greedy initial assignment
        # Sort tasks by length (2 first), then by conflict degree and class size
        tasks_sorted = list(tasks.items())
        random.shuffle(tasks_sorted)
        tasks_sorted.sort(key=lambda x: (-x[1], -conflict_degree[x[0]], -len(class_students[x[0]])))

        for cls, length in tasks_sorted:
            tchr = class_teacher[cls]
            best_pos = None
            best_inc = None

            if length == 1:
                for i in range(len(slots_by_day[d])):
                    if i in teacher_occ[tchr]:
                        continue
                    inc = sum(
                        1 for s in class_students[cls]
                        if student_slot_count[s][i] == 1
                    )
                    if best_inc is None or inc < best_inc:
                        best_inc = inc
                        best_pos = i
                if best_pos is None:
                    success = False
                    break
                assignments[cls] = best_pos
                teacher_occ[tchr].add(best_pos)
                for s in class_students[cls]:
                    student_slot_count[s][best_pos] += 1

            else:  # length == 2
                for i in range(len(slots_by_day[d]) - 1):
                    if i in teacher_occ[tchr] or (i + 1) in teacher_occ[tchr]:
                        continue
                    inc = sum(
                        1 for s in class_students[cls]
                        if student_slot_count[s][i] == 1 or student_slot_count[s][i + 1] == 1
                    )
                    if best_inc is None or inc < best_inc:
                        best_inc = inc
                        best_pos = i
                if best_pos is None:
                    success = False
                    break
                assignments[cls] = best_pos
                teacher_occ[tchr].add(best_pos)
                teacher_occ[tchr].add(best_pos + 1)
                for s in class_students[cls]:
                    student_slot_count[s][best_pos] += 1
                    student_slot_count[s][best_pos + 1] += 1

        if not success:
            break

        # Local search: move classes to reduce collisions (limited passes)
        # Build slots -> classes mapping
        slots_classes: dict[int, list[str]] = defaultdict(list)
        for cls, start in assignments.items():
            length = tasks[cls]
            if length == 1:
                slots_classes[start].append(cls)
            else:
                slots_classes[start].append(cls)
                slots_classes[start + 1].append(cls)

        def day_cost(slots_map):
            # Count students with >= 2 classes at any slot
            stud_counts = {s: defaultdict(int) for s in student_list}
            collisions = 0
            for slot, cls_list in slots_map.items():
                for cc in cls_list:
                    for s in class_students[cc]:
                        stud_counts[s][slot] += 1
            for s in student_list:
                if any(v >= 2 for v in stud_counts[s].values()):
                    collisions += 1
            return collisions

        current_cost = day_cost(slots_classes)
        for _ in range(3):  # number of local search passes
            improved = False
            for cls in list(assignments.keys()):
                length = tasks[cls]
                tchr = class_teacher[cls]
                current_pos = assignments[cls]

                # remove cls from current assignment
                if length == 1:
                    teacher_occ[tchr].remove(current_pos)
                    slots_classes[current_pos].remove(cls)
                    if not slots_classes[current_pos]:
                        del slots_classes[current_pos]
                else:
                    teacher_occ[tchr].remove(current_pos)
                    teacher_occ[tchr].remove(current_pos + 1)
                    slots_classes[current_pos].remove(cls)
                    if not slots_classes[current_pos]:
                        del slots_classes[current_pos]
                    slots_classes[current_pos + 1].remove(cls)
                    if not slots_classes[current_pos + 1]:
                        del slots_classes[current_pos + 1]

                best = current_pos
                best_cost = current_cost

                if length == 1:
                    for i in range(len(slots_by_day[d])):
                        if i in teacher_occ[tchr]:
                            continue
                        # add temporarily
                        slots_classes[i].append(cls)
                        teacher_occ[tchr].add(i)
                        new_cost = day_cost(slots_classes)
                        teacher_occ[tchr].remove(i)
                        slots_classes[i].remove(cls)
                        if not slots_classes[i]:
                            del slots_classes[i]
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best = i
                else:  # length == 2
                    for i in range(len(slots_by_day[d]) - 1):
                        if i in teacher_occ[tchr] or (i + 1) in teacher_occ[tchr]:
                            continue
                        slots_classes[i].append(cls)
                        slots_classes[i + 1].append(cls)
                        teacher_occ[tchr].add(i)
                        teacher_occ[tchr].add(i + 1)
                        new_cost = day_cost(slots_classes)
                        teacher_occ[tchr].remove(i)
                        teacher_occ[tchr].remove(i + 1)
                        slots_classes[i].remove(cls)
                        slots_classes[i + 1].remove(cls)
                        if not slots_classes[i]:
                            del slots_classes[i]
                        if not slots_classes[i + 1]:
                            del slots_classes[i + 1]
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best = i

                # apply best position
                if length == 1:
                    slots_classes[best].append(cls)
                    teacher_occ[tchr].add(best)
                else:
                    slots_classes[best].append(cls)
                    slots_classes[best + 1].append(cls)
                    teacher_occ[tchr].add(best)
                    teacher_occ[tchr].add(best + 1)
                assignments[cls] = best
                if best_cost < current_cost:
                    current_cost = best_cost
                    improved = True
            if not improved:
                break

        # Add final assignments for this day to the schedule
        for cls, start in assignments.items():
            length = tasks[cls]
            if length == 1:
                schedule[cls].append(slots_by_day[d][start])
            else:
                schedule[cls].append(slots_by_day[d][start])
                schedule[cls].append(slots_by_day[d][start + 1])

    # Check final schedule for this run
    if success:
        teacher_conf, student_clashes = evaluate(schedule)
        if teacher_conf == 0 and student_clashes < best_clashes:
            best_schedule = schedule
            best_clashes = student_clashes

# Write out the best schedule found
if best_schedule is not None:
    rows = []
    for cls, ts_list in best_schedule.items():
        for ts in ts_list:
            rows.append((ts, cls))

    # Sort for readability (by day, then time)
    def sort_key(row):
        ts = row[0]
        d = timeslot_days[ts]
        return (days_order.index(d), slots_by_day[d].index(ts))

    rows.sort(key=sort_key)
    with open('solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timeslot', 'class'])
        writer.writerows(rows)
    print(f"Schedule written with {best_clashes} student clashes.")
else:
    print("No valid schedule found.")

