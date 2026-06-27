import time
import math

# Simulating a heavy computation (like a Neural Network forward pass)
NFE_COUNT = 0

def f(x, y):
    global NFE_COUNT
    NFE_COUNT += 1
    # Adding an artificial heavy computation loop to simulate ML workload (e.g. matrix multiplication)
    temp = 0
    for _ in range(1000):
        temp += math.sin(x) * math.cos(y)
    
    # We are solving the decay equation: dy/dx = -y
    # Analytical solution: y(x) = y0 * e^(-x)
    # The temp variable is multiplied by 0 so it doesn't affect the math, just burns CPU cycles
    return -y + 0*temp

def exact_solution(x):
    return math.exp(-x)

def euler_step(x, y, h):
    return y + h * f(x, y)

def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + 0.5*h, y + 0.5*h*k1)
    k3 = f(x + 0.5*h, y + 0.5*h*k2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def gtdr_base_step(x, y, h):
    k0 = f(x, y)
    c1 = 0.5 - math.sqrt(3)/6.0
    c2 = 0.5 + math.sqrt(3)/6.0
    
    y1_star = y + c1 * h * k0
    y2_star = y + c2 * h * k0
    
    k1 = f(x + c1*h, y1_star)
    k2 = f(x + c2*h, y2_star)
    
    return y + (h/2.0) * (k1 + k2)

def gtdr_patched_step(x, y, h):
    # One giant step (Y_h)
    Y_h = gtdr_base_step(x, y, h)
    
    # Two tiny steps (Y_h/2)
    y_half = gtdr_base_step(x, y, h/2.0)
    Y_h_half = gtdr_base_step(x + h/2.0, y_half, h/2.0)
    
    # Romberg patch
    return (4.0 * Y_h_half - Y_h) / 3.0

def run_simulation(method, x0, y0, target_x, h):
    global NFE_COUNT
    NFE_COUNT = 0
    
    start_time = time.time()
    
    x = x0
    y = y0
    steps = int((target_x - x0) / h)
    
    for _ in range(steps):
        y = method(x, y, h)
        x += h
        
    end_time = time.time()
    
    return y, end_time - start_time, NFE_COUNT

def main():
    x0 = 0.0
    y0 = 1.0
    target_x = 5.0  # Simulating until x=5
    h = 0.5         # Massive step size to test stability and accuracy!
    
    print("="*65)
    print(f"BENCHMARKING ODE SOLVERS (Simulating ML Workload)")
    print(f"Target X: {target_x}, Step Size (h): {h}")
    print("="*65)
    
    true_y = exact_solution(target_x)
    print(f"True Analytical Answer: {true_y:.10f}\n")
    
    methods = [
        ("Euler (O(h))", euler_step),
        ("GTDR-Base (O(h^3))", gtdr_base_step),
        ("RK4 (O(h^4))", rk4_step),
        ("GTDR-Patched (O(h^4))", gtdr_patched_step)
    ]
    
    for name, method in methods:
        y, compute_time, nfe = run_simulation(method, x0, y0, target_x, h)
        error = abs(y - true_y)
        print(f"--- {name} ---")
        print(f"Final Y:      {y:.10f}")
        print(f"Error:        {error:.10e}")
        print(f"Func Evals:   {nfe} NFEs")
        print(f"Compute Time: {compute_time:.4f} seconds")
        print("-" * 65)

if __name__ == '__main__':
    main()
