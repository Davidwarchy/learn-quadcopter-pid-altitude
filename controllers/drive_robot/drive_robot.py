# drive_robot.py
import random
import numpy as np
import csv
from datetime import datetime
from maze_env import DroneRobot

timestamp = datetime.now().strftime('%Y%m%d-%H-%M-%S')

def genetic_algorithm(pop_size=20, generations=250, csv_filename=f"alt_fitness_log_{timestamp}.csv"):
    drone = DroneRobot()
    
    # Initialize population with altitude PID coefficients
    population = []
    for _ in range(pop_size):
        alt_pid_coeffs = [
            random.uniform(0.1, 5.0),  # k_alt_p
            random.uniform(0, 1.0),    # k_alt_i
            random.uniform(0, 2.0)     # k_alt_d
        ]
        population.append(alt_pid_coeffs)
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Generation', 'Individual', 'Fitness', 'k_alt_p', 'k_alt_i', 'k_alt_d', 'Timestamp'])
        
        for generation in range(generations):
            print(f"Progress: Starting generation {generation + 1}/{generations}")
            
            # Evaluate fitness
            fitness_scores = []
            for i, coeffs in enumerate(population):
                print(f'{i} Coeffs: {coeffs}')
                fitness = drone.simulate(coeffs)
                fitness_scores.append(fitness)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([generation + 1, i + 1, fitness, coeffs[0], coeffs[1], coeffs[2], timestamp])
                csvfile.flush()
            
            max_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            print(f"Generation {generation + 1}: Max Fitness = {max_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
            
            # Elitism
            elite_size = int(pop_size * 0.2)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite = [population[i] for i in sorted_indices[:elite_size]]
            
            # New population
            new_population = elite.copy()
            while len(new_population) < pop_size:
                parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
                
                # Crossover
                child = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]
                
                # Mutation
                for i in range(3):
                    if random.random() < 0.2:
                        child[i] += random.uniform(-0.1 * child[i], 0.1 * child[i])
                        child[i] = max(0, child[i])
                
                new_population.append(child)
            
            population = new_population[:pop_size]
        
        # Final evaluation
        final_fitness_scores = []
        for i, coeffs in enumerate(population):
            fitness = drone.simulate(coeffs)
            final_fitness_scores.append(fitness)
        
        best_index = np.argmax(final_fitness_scores)
        best_coeffs = population[best_index]
        
        print(f"\nBest Altitude PID Coefficients: k_alt_p={best_coeffs[0]:.2f}, "
              f"k_alt_i={best_coeffs[1]:.2f}, k_alt_d={best_coeffs[2]:.2f}")
        print(f"Best Fitness: {final_fitness_scores[best_index]:.2f}")
    
    return best_coeffs

if __name__ == "__main__":
    optimal_alt_pid = genetic_algorithm()