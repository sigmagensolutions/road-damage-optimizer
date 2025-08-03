import random
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from evaluate_thresholds import evaluate_thresholds

# GA hyperparameters
POPULATION_SIZE = 30
GENERATIONS = 50
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3

NUM_CLASSES = 4
LOW = 0.0
HIGH = 1.0
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

def random_thresholds():
    return [random.uniform(LOW, HIGH) for _ in range(NUM_CLASSES)]

def mutate(thresholds):
    return [
        min(max(t + random.gauss(0, 0.1), LOW), HIGH)
        if random.random() < MUTATION_RATE else t
        for t in thresholds
    ]

def crossover(parent1, parent2):
    return [random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)]

def tournament_selection(population, scores):
    selected = random.sample(list(zip(population, scores)), TOURNAMENT_SIZE)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

def genetic_algorithm():
    population = [random_thresholds() for _ in range(POPULATION_SIZE)]
    best_solution = None
    best_score = float("inf")
    history = []

    for gen in range(GENERATIONS):
        scores = [evaluate_thresholds(ind) for ind in population]
        gen_best_idx = np.argmin(scores)
        gen_best = population[gen_best_idx]
        gen_best_score = scores[gen_best_idx]

        history.append(gen_best_score)

        if gen_best_score < best_score:
            best_score = gen_best_score
            best_solution = gen_best

        print(f"Gen {gen+1:02d} | Best Score: ${best_score:.2f} | Thresholds: {np.round(best_solution, 3).tolist()}")

        new_population = [best_solution]  # elitism
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            child = mutate(crossover(parent1, parent2))
            new_population.append(child)

        population = new_population

    # Save results
    save_results(best_solution, best_score, history)

    return best_solution, best_score

def save_results(best_solution, best_score, history):
    rounded = np.round(best_solution, 4).tolist()

    # Save JSON
    with open("best_thresholds.json", "w") as f:
        json.dump({
            "thresholds": dict(zip(CLASS_NAMES, rounded)),
            "cost": best_score
        }, f, indent=2)

    # Save CSV
    pd.DataFrame({
        "Generation": list(range(1, len(history)+1)),
        "Best_Cost": history
    }).to_csv("cost_history.csv", index=False)

    # Plot
    plt.plot(history, label="Best cost")
    plt.xlabel("Generation")
    plt.ylabel("Cost ($)")
    plt.title("Cost Evolution over Generations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cost_evolution.png")
    plt.show()

    print("\n📁 Saved:")
    print("- best_thresholds.json")
    print("- cost_history.csv")
    print("- cost_evolution.png")

if __name__ == "__main__":
    genetic_algorithm()
