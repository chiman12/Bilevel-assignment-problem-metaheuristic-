!pip install -U mealpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.physics_based.SA import OriginalSA

from mealpy.utils.problem import Problem
from mealpy.utils.space import IntegerVar


# ==============================
# 🔹 Lower-level objective: patients -> beds in one hospital
# ==============================
def lower_objective(patients_in_hospital, consult_time, assign_time, n_beds):
    """
    Optimize allocation of patients to beds inside a hospital.
    A simple greedy model: if patients > beds, some patients wait longer.
    """
    n_patients = len(patients_in_hospital)
    if n_patients == 0:
        return 0

    # Each patient has a consultation + administrative time
    base_time = (consult_time + assign_time) * n_patients

    # Penalty if not enough beds
    if n_patients > n_beds:
        penalty = (n_patients - n_beds) * (consult_time + assign_time) * 2
    else:
        penalty = 0

    return base_time + penalty


# ==============================
# 🔹 Upper-level objective: patients -> hospitals
# ==============================
def bilevel_objective(solution, travel, waiting, consult, assign, hospital_caps, beds):
    """
    Bi-level objective:
    - Upper: patient -> hospital (travel + waiting + hospital capacity)
    - Lower: patient -> bed inside hospital (consult + assign + penalty if overflow)
    """
    solution = solution.astype(int)
    total_time = 0

    # Upper level contribution
    for i, hospital_idx in enumerate(solution):
        total_time += travel[i, hospital_idx] + waiting[hospital_idx]

    # Lower level contribution
    for h in range(len(hospital_caps)):
        patients_h = [i for i, s in enumerate(solution) if s == h]
        total_time += lower_objective(patients_h, consult[h], assign[h], beds[h])

    return total_time


# ==============================
# 🔹 Run metaheuristic algorithm
# ==============================
def run_algo(AlgoClass, n_patients, n_hospitals, travel, waiting, consult, assign, hospital_caps, beds):
    def fitness(solution):
        solution = np.clip(np.round(solution).astype(int), 0, n_hospitals - 1)
        return bilevel_objective(solution, travel, waiting, consult, assign, hospital_caps, beds)

    bounds = [IntegerVar(0, n_hospitals - 1) for _ in range(n_patients)]

    class HospitalAssignmentProblem(Problem):
        def __init__(self, fit_func, bounds):
            super().__init__(fit_func=fit_func, bounds=bounds, minimize=True, name="HospitalAssignment")

        def obj_func(self, solution):
            return self.fit_func(solution)

    problem = HospitalAssignmentProblem(fitness, bounds)

    model = AlgoClass(epoch=100, pop_size=30)
    start = time.time()
    best_agent = model.solve(problem)
    end = time.time()

    return best_agent.solution, best_agent.target.fitness, end - start


# ==============================
# 🔹 Dynamic simulation setup
# ==============================
n_periods = 5
patients_init = 200
hospitals_init = 5
patient_growth = 0.05
hospital_growth = 0.02

patients_list, hospitals_list = [], []
p, h = patients_init, hospitals_init

for _ in range(n_periods):
    p = max(1, int(p * (1 + patient_growth + np.random.normal(0, 0.02))))
    h = max(1, int(h * (1 + hospital_growth + np.random.normal(0, 0.01))))
    patients_list.append(p)
    hospitals_list.append(h)


# ==============================
# 🔹 Data generation
# ==============================
def generate_times(n_patients, n_hospitals):
    travel = np.random.randint(5, 60, size=(n_patients, n_hospitals))
    waiting = np.random.randint(10, 120, size=n_hospitals)
    consult = np.random.randint(15, 30, size=n_hospitals)
    assign = np.random.randint(5, 30, size=n_hospitals)
    hospital_caps = np.random.randint(20, 80, size=n_hospitals)  # hospital patient capacity
    beds = np.random.randint(10, 50, size=n_hospitals)           # number of beds per hospital
    return travel, waiting, consult, assign, hospital_caps, beds


# ==============================
# 🔹 Algorithms
# ==============================
algos = {
    "GA": BaseGA,
    "PSO": OriginalPSO,
    "SA": OriginalSA,
    "ABC": OriginalABC
}

results = []


# ==============================
# 🔹 Main loop
# ==============================
for period in range(n_periods):
    n_patients = patients_list[period]
    n_hospitals = hospitals_list[period]
    travel, waiting, consult, assign, hospital_caps, beds = generate_times(n_patients, n_hospitals)

    print(f"\n🗓️ Period {period+1}: {n_patients} patients, {n_hospitals} hospitals")

    for name, algo_class in algos.items():
        best_solution, best_fit, duration = run_algo(
            algo_class, n_patients, n_hospitals, travel, waiting, consult, assign, hospital_caps, beds
        )
        print(f"  {name} => Best total time: {best_fit:.2f} min | Duration: {duration:.2f} s")
        results.append({
            "Period": period + 1,
            "Algorithm": name,
            "Patients": n_patients,
            "Hospitals": n_hospitals,
            "BestTime": best_fit,
            "Duration": duration
        })


# ==============================
# 🔹 Visualization
# ==============================
df_results = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
for name in algos:
    subset = df_results[df_results["Algorithm"] == name]
    plt.plot(subset["Period"], subset["BestTime"], marker='o', label=name)

plt.title("Bilevel optimal total time per algorithm and period")
plt.xlabel("Period")
plt.ylabel("Total time (minutes)")
plt.legend()
plt.grid(True)
plt.show()


