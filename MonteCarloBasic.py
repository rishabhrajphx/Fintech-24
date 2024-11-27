import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(num_points=10000):
    # Generate random points
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)

    # Calculate distance from origin
    distances = np.sqrt(x**2 + y**2)

    # Count points inside the circle
    points_inside = np.sum(distances <= 1)

    # Calculate pi: (points inside circle / total points) * 4
    pi_estimate = 4 * points_inside / num_points

    return pi_estimate, x, y, distances

def visualize_simulation(x, y, distances):
    plt.figure(figsize=(10, 10))

    # Plot points inside circle in blue
    plt.scatter(x[distances <= 1], y[distances <= 1], c='blue', alpha=0.6, label='Inside')

    # Plot points outside circle in red
    plt.scatter(x[distances > 1], y[distances > 1], c='red', alpha=0.6, label='Outside')

    # Draw the circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    plt.gca().add_artist(circle)

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Monte Carlo Simulation for π Estimation')
    plt.show()

def main():
    # Run simulation
    num_points = 100000
    estimated_pi, x, y, distances = estimate_pi(num_points)

    # Print results
    print(f"Estimated π: {estimated_pi:.6f}")
    print(f"Actual π: {np.pi:.6f}")
    print(f"Difference: {abs(estimated_pi - np.pi):.6f}")

    # Visualize results
    visualize_simulation(x, y, distances)

if __name__ == "__main__":
    main()