# Timetable Generator

## Overview

Timetable Generator is a Python-based university classroom scheduling tool that leverages Constraint Satisfaction Problems (CSP) and Genetic Algorithms (GA) to create feasible, balanced, and efficient timetables for courses, instructors, and classrooms. The system is designed to handle complex constraints and optimize schedules for both regular classes and lab sessions.

## Features

- **Constraint Satisfaction Problem (CSP) Feasibility Check**: Ensures that a valid timetable is possible given the courses, instructors, and classroom constraints.
- **Genetic Algorithm Optimization**: Uses evolutionary techniques to generate and improve timetables, minimizing gaps and balancing instructor workloads.
- **Flexible Layouts**: Supports multiple daily period layouts (including breaks and lab slots) and alternates layouts across days.
- **Detailed Statistics**: Provides day-wise session distribution, gap analysis, and instructor workload summaries.
- **Customizable Inputs**: Easily modify courses, instructors, classroom names, and period layouts in the code.

## How It Works

1. **Define Inputs**: Specify courses (with credit hours), instructors (with assigned courses), classroom name, and daily period layouts.
2. **CSP Feasibility**: The scheduler first checks if a feasible timetable exists using CSP techniques.
3. **Genetic Algorithm Scheduling**: If feasible, a genetic algorithm generates and evolves candidate timetables, optimizing for constraints and preferences.
4. **Output**: The final timetable, statistics, and instructor schedules are printed to the console.

## Requirements

Install dependencies with:

```bash
pip install python-constraint deap
```

## Usage

1. Clone or download this repository.
2. Edit `scheduler.py` to customize courses, instructors, layouts, or days as needed.
3. Run the scheduler:

```bash
python scheduler.py
```

4. The script will print:
   - Feasibility status
   - The generated timetable for each day
   - Schedule statistics (sessions per day, gaps, instructor workload)
   - Individual instructor schedules

---

## Project Documents

- [Similarity Report](./similarity_report.pdf)
- [Submission Receipt](./submission_reciept.pdf)
- [Project Report](./timetable-generator-5.pdf)

---

**Disclaimer:**
This repository and its documents are intended as project reports for academic or demonstration purposes only. They are not formal research papers and should not be cited as such.
