# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pylint: disable=broad-except

import random
import time
from collections import defaultdict

from constraint import AllDifferentConstraint, Problem
from deap import algorithms, base, creator, tools


class ClassroomScheduler:
    def __init__(
        self, courses, instructors, classroom_name, layouts, day_layout_map, days
    ):
        self.courses = courses
        self.instructors = instructors
        self.classroom_name = classroom_name
        self.layouts = layouts
        self.days = days
        self.day_layouts = day_layout_map

        # Calculate sessions needed per course
        self.course_sessions = {
            course["code"]: (
                1 if "Lab" in course["code"] else min(3, course["credit_hours"])
            )
            for course in self.courses
        }

        # Generate available slots and teaching periods
        self.available_slots = []
        self.teaching_periods = {}
        for day in self.days:
            layout = self.day_layouts[day]
            self.teaching_periods[day] = [
                pid for pid, name in layout.items() if "Break" not in name
            ]
            for period_id, period_name in layout.items():
                if "Break" not in period_name:
                    self.available_slots.append((day, period_id))

        self.target_load_per_day = sum(self.course_sessions.values()) / len(self.days)

        # Instructor mappings
        self.course_instructor_map = {}
        self.instructor_courses = defaultdict(list)
        for instructor in self.instructors:
            for course in instructor.get("assigned_courses", []):
                self.course_instructor_map[course] = instructor["name"]
                self.instructor_courses[instructor["name"]].append(course)

    def is_lab_period(self, day, period_id):
        return "Lab" in self.day_layouts[day][period_id]

    def check_feasibility_with_csp(self):
        problem = Problem()
        variables = []

        for course in self.courses:
            code = course["code"]
            sessions = self.course_sessions[code]

            for i in range(sessions):
                var_name = f"{code}_{i}"
                variables.append(var_name)

                # Domain based on course type
                is_lab = "Lab" in code
                domain = [
                    (day, pid)
                    for day, pid in self.available_slots
                    if (is_lab and self.is_lab_period(day, pid))
                    or (not is_lab and not self.is_lab_period(day, pid))
                ]

                if not domain:
                    domain = self.available_slots

                problem.addVariable(var_name, domain)

        # Constraints
        problem.addConstraint(AllDifferentConstraint(), variables)

        for course in self.courses:
            code = course["code"]
            session_vars = [f"{code}_{i}" for i in range(self.course_sessions[code])]

            if len(session_vars) > 1:
                problem.addConstraint(
                    lambda *slots: len(set(slot[0] for slot in slots)) == len(slots),
                    session_vars,
                )

        try:
            return bool(problem.getSolution())
        except Exception:
            return False

    def _calculate_day_gap_score(self, day, periods):
        if not periods:
            return 0

        periods = sorted(periods)
        gap_score = 0
        teaching_periods = self.teaching_periods[day]

        for i in range(len(periods) - 1):
            current, next_class = periods[i], periods[i + 1]
            gap_size = sum(1 for p in teaching_periods if current < p < next_class)
            if gap_size > 0:
                gap_score += gap_size * 20

        return gap_score

    def generate_initial_schedule(self):
        schedule = []
        used_slots = set()
        course_days = defaultdict(set)
        day_load = {day: 0 for day in self.days}

        # Group slots by day
        slots_by_day = defaultdict(list)
        for day, period_id in self.available_slots:
            slots_by_day[day].append(period_id)
        for day in slots_by_day:
            slots_by_day[day].sort()

        # Process regular courses first, then labs
        for course_group in [
            [c for c in self.courses if "Lab" not in c["code"]],
            [c for c in self.courses if "Lab" in c["code"]],
        ]:
            random.shuffle(course_group)

            for course in course_group:
                code = course["code"]
                for _ in range(self.course_sessions[code]):
                    best_slot, best_day, best_score = None, None, float("inf")

                    # Find the best day and slot
                    for day in self.days:
                        if day in course_days[code]:
                            continue

                        day_schedule = [p for d, p in used_slots if d == day]

                        for period_id in slots_by_day[day]:
                            if (day, period_id) in used_slots:
                                continue

                            is_lab_course = "Lab" in code
                            is_lab_period = self.is_lab_period(day, period_id)
                            if (is_lab_course and not is_lab_period) or (
                                not is_lab_course and is_lab_period
                            ):
                                continue

                            # Calculate scores
                            potential_day_schedule = sorted(day_schedule + [period_id])
                            gap_score = self._calculate_day_gap_score(
                                day, potential_day_schedule
                            )

                            potential_day_load = day_load.copy()
                            potential_day_load[day] += 1
                            balance_score = sum(
                                abs(count - self.target_load_per_day) * 10
                                for _, count in potential_day_load.items()
                            )

                            combined_score = gap_score * 3 + balance_score

                            if combined_score < best_score:
                                best_score, best_day, best_slot = (
                                    combined_score,
                                    day,
                                    period_id,
                                )

                    # Schedule the slot
                    if best_day and best_slot:
                        instructor = self.course_instructor_map.get(code, "Unassigned")
                        schedule.append((code, best_day, best_slot, instructor))
                        used_slots.add((best_day, best_slot))
                        course_days[code].add(best_day)
                        day_load[best_day] += 1
                    else:
                        # Fallback to any available slot
                        for day, period_id in self.available_slots:
                            if (day, period_id) not in used_slots:
                                is_lab_course = "Lab" in code
                                is_lab_period = self.is_lab_period(day, period_id)
                                if (is_lab_course and is_lab_period) or (
                                    not is_lab_course and not is_lab_period
                                ):
                                    instructor = self.course_instructor_map.get(
                                        code, "Unassigned"
                                    )
                                    schedule.append((code, day, period_id, instructor))
                                    used_slots.add((day, period_id))
                                    course_days[code].add(day)
                                    day_load[day] += 1
                                    break

        return schedule

    def solve_with_genetic_algorithm(self):
        # Clean up previous DEAP classes if they exist
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register(
            "individual", lambda: creator.Individual(self.generate_initial_schedule())
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function
        def evaluate(schedule):
            penalty = 0
            course_counts, course_days = defaultdict(int), defaultdict(set)

            for course, day, _, _ in schedule:
                course_counts[course] += 1
                course_days[course].add(day)

            # Check required sessions
            for course in self.courses:
                code = course["code"]
                required = self.course_sessions[code]
                scheduled = course_counts.get(code, 0)

                if scheduled < required:
                    penalty += 100 * (required - scheduled)
                elif scheduled > required:
                    penalty += 50 * (scheduled - required)

            # Check for same-day sessions
            for course, days in course_days.items():
                if course in course_counts and course_counts[course] > len(days):
                    penalty += 30 * (course_counts[course] - len(days))

            # Check for double-booked slots and period compatibility
            slot_usage = {}
            day_load, days_schedule = defaultdict(int), defaultdict(list)

            for course, day, period_id, _ in schedule:
                slot = (day, period_id)
                if slot in slot_usage:
                    penalty += 200  # Double booking
                slot_usage[slot] = course

                # Check period type compatibility
                is_lab_course = "Lab" in course
                is_lab_period = self.is_lab_period(day, period_id)
                if is_lab_course and not is_lab_period:
                    penalty += 20
                elif not is_lab_course and is_lab_period:
                    penalty += 10

                day_load[day] += 1
                days_schedule[day].append(period_id)

            # Add balance and gap penalties
            penalty += sum(
                abs(count - self.target_load_per_day) * 10
                for day, count in day_load.items()
            )
            for day, periods in days_schedule.items():
                penalty += self._calculate_day_gap_score(day, periods) * 4

            return (penalty,)

        def crossover(ind1, ind2):
            if not ind1 or not ind2:
                return ind1, ind2
            try:
                crosspoint = random.randint(1, min(len(ind1), len(ind2)) - 1)
                ind1[crosspoint:], ind2[crosspoint:] = (
                    ind2[crosspoint:],
                    ind1[crosspoint:],
                )
                return ind1, ind2
            except Exception:
                return ind1, ind2

        def mutate(individual, indpb):
            if not individual:
                return (individual,)

            try:
                # Calculate current state
                day_load, day_schedules, used_slots = (
                    defaultdict(int),
                    defaultdict(list),
                    set(),
                )

                for _, day, period_id, _ in individual:
                    day_load[day] += 1
                    day_schedules[day].append(period_id)
                    used_slots.add((day, period_id))

                for day in day_schedules:
                    day_schedules[day].sort()

                most_loaded = max(day_load.items(), key=lambda x: x[1])[0]
                least_loaded = min(day_load.items(), key=lambda x: x[1])[0]
                available_slots = [
                    (day, pid)
                    for day, pid in self.available_slots
                    if (day, pid) not in used_slots
                ]

                for i in range(len(individual)):
                    if random.random() < indpb:
                        course, day, period_id, instructor = individual[i]
                        mutation_type = random.choice(
                            ["reduce_gaps", "reduce_gaps", "rebalance"]
                        )

                        if (
                            mutation_type == "rebalance"
                            and day == most_loaded
                            and least_loaded
                            not in [d for c, d, _, _ in individual if c == course]
                        ):
                            suitable_slots = [
                                (d, p)
                                for d, p in available_slots
                                if d == least_loaded
                                and (("Lab" in course) == self.is_lab_period(d, p))
                            ]

                            if suitable_slots:
                                new_day, new_period = random.choice(suitable_slots)
                                available_slots.remove((new_day, new_period))
                                available_slots.append((day, period_id))
                                individual[i] = (
                                    course,
                                    new_day,
                                    new_period,
                                    instructor,
                                )
                                day_schedules[day].remove(period_id)
                                day_schedules[new_day].append(new_period)
                                day_schedules[new_day].sort()

                        elif mutation_type == "reduce_gaps":
                            day_periods = day_schedules[day].copy()
                            if period_id in day_periods:
                                day_periods.remove(period_id)

                            day_available = [
                                (d, p) for d, p in available_slots if d == day
                            ]

                            if day_available:
                                best_period, min_gap_score = None, float("inf")

                                for _, potential_period in day_available:
                                    if ("Lab" in course) != self.is_lab_period(
                                        day, potential_period
                                    ):
                                        continue

                                    test_periods = sorted(
                                        day_periods + [potential_period]
                                    )
                                    gap_score = self._calculate_day_gap_score(
                                        day, test_periods
                                    )

                                    if gap_score < min_gap_score:
                                        min_gap_score = gap_score
                                        best_period = potential_period

                                if best_period:
                                    individual[i] = (
                                        course,
                                        day,
                                        best_period,
                                        instructor,
                                    )
                                    used_slots.remove((day, period_id))
                                    used_slots.add((day, best_period))
                                    day_schedules[day].remove(period_id)
                                    day_schedules[day].append(best_period)
                                    day_schedules[day].sort()

                return (individual,)
            except Exception:
                return (individual,)

        # Register operators
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Run the algorithm
        start_time = time.time()
        try:
            population = toolbox.population(n=50)
            hall_of_fame = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", min)

            for _ in range(25):
                if time.time() - start_time > 60:  # 60 seconds max
                    break

                offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
                fits = toolbox.map(toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit

                population = toolbox.select(offspring, len(population))
                hall_of_fame.update(population)

            return self.format_timetable(
                hall_of_fame[0] if hall_of_fame else self.generate_initial_schedule()
            )
        except Exception:
            return self.format_timetable(self.generate_initial_schedule())

    def format_timetable(self, schedule):
        timetable = {
            day: {pid: None for pid in self.day_layouts[day]} for day in self.days
        }
        for course, day, period_id, instructor in schedule:
            timetable[day][period_id] = {"course": course, "instructor": instructor}
        return timetable

    def solve(self):
        is_feasible = self.check_feasibility_with_csp()
        timetable = self.solve_with_genetic_algorithm()

        # Format results for backend integration
        result = {
            "is_feasible": is_feasible,
            "metadata": {"classroom": self.classroom_name},
            "timetable": {},
            "instructor_schedules": self._get_instructor_schedules_dict(timetable),
            "stats": self._get_schedule_stats(timetable),
        }

        # Format timetable
        for day in self.days:
            result["timetable"][day] = {
                "layout": 1 if self.day_layouts[day] == self.layouts["layout1"] else 2,
                "periods": {},
            }

            for period_id, period_name in self.day_layouts[day].items():
                period_data = {
                    "time": period_name,
                    "is_break": "Break" in period_name,
                    "course": None,
                    "instructor": None,
                    "is_lab": None,
                }

                if timetable[day][period_id] is not None:
                    period_data.update(
                        {
                            "course": timetable[day][period_id]["course"],
                            "instructor": timetable[day][period_id]["instructor"],
                            "is_lab": "Lab" in timetable[day][period_id]["course"],
                        }
                    )

                result["timetable"][day]["periods"][period_id] = period_data

        return result

    def _get_schedule_stats(self, timetable):
        # Day load distribution
        day_load = {
            day: sum(
                1
                for pid, entry in timetable[day].items()
                if entry is not None and "Break" not in self.day_layouts[day][pid]
            )
            for day in self.days
        }

        # Calculate statistics
        total_sessions = sum(day_load.values())
        avg_per_day = total_sessions / len(self.days)
        std_dev = (
            sum((count - avg_per_day) ** 2 for count in day_load.values())
            / len(self.days)
        ) ** 0.5

        # Analyze gaps
        gaps, total_gaps = {}, 0
        for day in self.days:
            layout = self.day_layouts[day]
            teaching_periods = self.teaching_periods[day]
            periods_in_use = sorted(
                [
                    pid
                    for pid, entry in timetable[day].items()
                    if entry is not None and "Break" not in layout[pid]
                ]
            )

            if len(periods_in_use) > 1:
                day_gaps = sum(
                    1
                    for i in range(len(periods_in_use) - 1)
                    for p in teaching_periods
                    if periods_in_use[i] < p < periods_in_use[i + 1]
                )

                if day_gaps > 0:
                    gaps[day] = day_gaps
                    total_gaps += day_gaps

        # Analyze instructor workload
        instructor_load, instructor_days = defaultdict(int), defaultdict(set)
        for day in self.days:
            for pid, entry in timetable[day].items():
                if entry is not None:
                    instructor = entry["instructor"]
                    instructor_load[instructor] += 1
                    instructor_days[instructor].add(day)

        instructor_stats = {
            instructor: {
                "sessions": count,
                "days": len(instructor_days[instructor]),
                "days_list": sorted(list(instructor_days[instructor])),
            }
            for instructor, count in instructor_load.items()
        }

        return {
            "day_distribution": day_load,
            "total_sessions": total_sessions,
            "avg_sessions_per_day": avg_per_day,
            "std_deviation": std_dev,
            "gaps": gaps,
            "total_gaps": total_gaps,
            "instructor_workload": instructor_stats,
        }

    def _get_instructor_schedules_dict(self, timetable):
        instructor_schedules = defaultdict(list)

        for day in self.days:
            layout = self.day_layouts[day]
            layout_idx = 1 if layout == self.layouts["layout1"] else 2

            for period_id, entry in timetable[day].items():
                if entry is not None:
                    instructor, course = entry["instructor"], entry["course"]
                    instructor_schedules[instructor].append(
                        {
                            "day": day,
                            "layout": layout_idx,
                            "period_id": period_id,
                            "time": layout[period_id],
                            "course": course,
                            "is_lab": "Lab" in course,
                        }
                    )

        # Add summary data
        for instructor in instructor_schedules:
            course_days = defaultdict(set)
            for entry in instructor_schedules[instructor]:
                course_days[entry["course"]].add(entry["day"])

            course_summary = {
                course: {"days_count": len(days), "days": sorted(list(days))}
                for course, days in course_days.items()
            }

            instructor_schedules[instructor].append(
                {
                    "is_summary": True,
                    "total_sessions": len(instructor_schedules[instructor]),
                    "unique_days": len(
                        set(
                            entry["day"]
                            for entry in instructor_schedules[instructor]
                            if "is_summary" not in entry
                        )
                    ),
                    "courses": course_summary,
                }
            )

        return dict(instructor_schedules)


def main():
    # Define layouts
    layouts = {
        "layout1": {
            1: "08:00-09:00",
            2: "09:00-10:00",
            3: "10:00-10:30(Break)",
            4: "10:30-11:30",
            5: "11:30-12:30",
            6: "12:30-14:00(Break)",
            7: "14:00-16:30(Lab)",
        },
        "layout2": {
            1: "9-11:30(Lab)",
            2: "11:30-13:00(Break)",
            3: "13:00-14:00",
            4: "14:00-15:00",
            5: "15-15:30(Break)",
            6: "15:30-16:30",
        },
    }

    # Assign layouts to days
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    day_layout_map = {}

    # Automatically alternate layouts for each day
    start_with_layout1 = random.choice([True, False])
    for i, day in enumerate(days):
        use_layout1 = start_with_layout1 if i % 2 == 0 else not start_with_layout1
        layout_key = "layout1" if use_layout1 else "layout2"
        day_layout_map[day] = layouts[layout_key]

    # Define courses and instructors
    courses = [
        {"code": "DAA", "credit_hours": 4},
        {"code": "DBS", "credit_hours": 4},
        {"code": "AI", "credit_hours": 4},
        {"code": "MAT", "credit_hours": 3},
        {"code": "OS", "credit_hours": 3},
        {"code": "DAA Lab", "credit_hours": 1},
        {"code": "DBS Lab", "credit_hours": 1},
        {"code": "AI Lab", "credit_hours": 1},
    ]

    instructors = [
        {"name": "Dr. Williams", "assigned_courses": ["DAA", "DAA Lab"]},
        {"name": "Prof. Johnson", "assigned_courses": ["AI", "AI Lab"]},
        {"name": "Dr. Smith", "assigned_courses": ["DBS", "DBS Lab"]},
        {"name": "Dr. Jane", "assigned_courses": ["OS"]},
        {"name": "Prof. Alex", "assigned_courses": ["MAT"]},
    ]

    # Create scheduler and run
    classroom_name = "Room A"
    scheduler = ClassroomScheduler(
        courses, instructors, classroom_name, layouts, day_layout_map, days
    )
    result = scheduler.solve()

    # Display all data
    print("\n" + "=" * 50)
    print(f"CLASSROOM SCHEDULE FOR: {result['metadata']['classroom']}")
    print("=" * 50)
    print(
        f"\nFeasibility Status: {'✓ Feasible' if result['is_feasible'] else '✗ Not Feasible'}"
    )

    # Display Timetable
    print("\n" + "=" * 50)
    print("TIMETABLE")
    print("=" * 50)

    for day in days:
        layout_num = result["timetable"][day]["layout"]
        print(f"\n{day} (Layout {layout_num}):")
        print("-" * 40)

        for period_id in sorted(result["timetable"][day]["periods"].keys()):
            period_data = result["timetable"][day]["periods"][period_id]
            time_str = period_data["time"]

            if period_data["is_break"]:
                print(f"  Period {period_id}: {time_str} [BREAK]")
            elif period_data["course"] is None:
                print(f"  Period {period_id}: {time_str} [FREE]")
            else:
                lab_indicator = " (Lab)" if period_data["is_lab"] else ""
                print(
                    f"  Period {period_id}: {time_str} - {period_data['course']}{lab_indicator} - {period_data['instructor']}"
                )

    # Display Statistics
    print("\n" + "=" * 50)
    print("SCHEDULE STATISTICS")
    print("=" * 50)

    stats = result["stats"]
    print("\nDay Distribution:")
    for day, count in stats["day_distribution"].items():
        print(f"  {day}: {count} sessions")

    print(f"\nTotal Sessions: {stats['total_sessions']}")
    print(f"Average Sessions Per Day: {stats['avg_sessions_per_day']:.2f}")
    print(f"Standard Deviation: {stats['std_deviation']:.2f}")

    # Gaps
    print("\nGaps in Schedule:")
    if stats["total_gaps"] == 0:
        print("  No gaps in the schedule")
    else:
        for day, gap_count in stats["gaps"].items():
            print(f"  {day}: {gap_count} gaps")
        print(f"Total Gaps: {stats['total_gaps']}")

    # Display Instructor Schedules
    print("\n" + "=" * 50)
    print("INSTRUCTOR SCHEDULES")
    print("=" * 50)

    for instructor, schedule in result["instructor_schedules"].items():
        summary = next((item for item in schedule if "is_summary" in item), None)
        regular_entries = [item for item in schedule if "is_summary" not in item]

        if summary:
            print(f"\n{instructor}:")
            print(f"  Total Sessions: {summary['total_sessions']}")
            print(f"  Teaching Days: {summary['unique_days']}")

            print("\n  Courses:")
            for course, course_data in summary["courses"].items():
                days_list = ", ".join(course_data["days"])
                print(f"    - {course}: {course_data['days_count']} days ({days_list})")

            print("\n  Detailed Schedule:")
            by_day = {}
            for entry in regular_entries:
                if entry["day"] not in by_day:
                    by_day[entry["day"]] = []
                by_day[entry["day"]].append(entry)

            for day in days:
                if day in by_day:
                    print(f"    {day}:")
                    for entry in sorted(by_day[day], key=lambda x: x["period_id"]):
                        lab_indicator = " (Lab)" if entry["is_lab"] else ""
                        print(
                            f"      Period {entry['period_id']}: {entry['time']} - {entry['course']}{lab_indicator}"
                        )

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
