Timetable test problem:

Your task is to create an optimal timetable - that is, an assignment of classes to time slots, given the input data listed below. You may use any method and language you like. The quality of the solution will be evaluated based on the following criteria:
1. Lesson spread: for each class, the lessons should be spread throughout the weekdays as evenly as possible 
2. Within the same day, if there are multiple lessons, they should be consecutive if possible.
3. Time clashes for teachers are not allowed at all.
4. Time clashes for students must be avoided where possible.
5. Each class should be assigned the required number of lessons exactly.

Input data:
- List of students
- List of teachers
- List of weekdays
- List of time slots, with each linked to one weekday
- List of classes
- Student class assignment
- Teacher class assignment
- Number of lessons required for each class, for the whole week

Output expected format:
The output should be in .csv format, with two columns: timeslot and class. A class and timeslot in the same row indicates that that timeslot is assigned to that class, i.e. there is a lesson of that class scheduled to occur at that time.

Testing:
Run python opt_test.py <your_solution_filename>
Make sure the input files are in the same directory as the test file.

