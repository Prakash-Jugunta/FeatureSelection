import numpy as np

def objective_function(x, y):
    return x**2 + y**2  

num_particles = 10
num_iterations = 100
w = 0.5 
c1, c2 = 1.5, 1.5 

particles = np.random.uniform(-10, 10, (num_particles, 2))
print("Particles")
print(particles)
velocities = np.random.uniform(-1, 1, (num_particles, 2))
print("Velocities")
print(velocities)

p_best_positions = particles.copy()
p_best_values = np.apply_along_axis(lambda p: objective_function(p[0], p[1]), 1, particles)
g_best_index = np.argmin(p_best_values)
g_best_position = p_best_positions[g_best_index]

for _ in range(num_iterations):
    for i in range(num_particles):
        fitness = objective_function(particles[i][0], particles[i][1])
        
        if fitness < p_best_values[i]:
            p_best_values[i] = fitness
            p_best_positions[i] = particles[i].copy()

    g_best_index = np.argmin(p_best_values)
    g_best_position = p_best_positions[g_best_index]

    r1, r2 = np.random.rand(num_particles, 1), np.random.rand(num_particles, 1)
    velocities = (w * velocities +
                  c1 * r1 * (p_best_positions - particles) +
                  c2 * r2 * (g_best_position - particles))
    particles += velocities
    
print("Optimal solution found at:", g_best_position)
print("Minimum function value:", objective_function(g_best_position[0], g_best_position[1]))
