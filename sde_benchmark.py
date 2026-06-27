import numpy as np
import time

def run_sde_benchmark():
    np.random.seed(42) # For a reproducible path
    
    # Financial Parameters (Geometric Brownian Motion)
    # This is the exact model used in the Black-Scholes formula
    S0 = 100.0   # Initial stock price ($100)
    mu = 0.10    # Expected return (10%)
    sigma = 0.30 # Volatility (30% - quite noisy!)
    T = 1.0      # 1 year simulation
    N = 50       # Number of steps (using a large step size to stress-test)
    h = T / N
    
    # Pre-generate the random Market Noise (Brownian Motion)
    # We force all methods to experience the EXACT same market shocks
    dW = np.random.normal(0, np.sqrt(h), N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0.0) # start at 0
    
    # 1. EXACT SOLUTION for Geometric Brownian Motion
    t_exact = np.linspace(0, T, N+1)
    S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * t_exact + sigma * W)
    
    # 2. EULER-MARUYAMA METHOD (The Industry Standard for simple SDEs)
    S_em = np.zeros(N+1)
    S_em[0] = S0
    for i in range(N):
        y_n = S_em[i]
        drift = mu * y_n
        diffusion = sigma * y_n
        S_em[i+1] = y_n + drift * h + diffusion * dW[i]
        
    # 3. NAIVE STOCHASTIC WILLIAMS FORMULA
    # We apply the high-order Williams Formula to the deterministic drift,
    # and add standard diffusion to see if it survives the noise.
    S_will = np.zeros(N+1)
    S_will[0] = S0
    c1 = 0.5 - np.sqrt(3)/6.0
    c2 = 0.5 + np.sqrt(3)/6.0
    
    for i in range(N):
        y_n = S_will[i]
        
        # Williams Drift Approximation
        k0 = mu * y_n
        y1_star = y_n + c1 * h * k0
        y2_star = y_n + c2 * h * k0
        
        k1 = mu * y1_star
        k2 = mu * y2_star
        
        # Final Williams Step + Stochastic Noise
        drift_term = (h / 2.0) * (k1 + k2)
        diffusion_term = sigma * y_n * dW[i]
        
        S_will[i+1] = y_n + drift_term + diffusion_term

    # Calculate Errors at the final step T=1.0
    exact_final = S_exact[-1]
    em_final = S_em[-1]
    will_final = S_will[-1]
    
    err_em = abs(em_final - exact_final)
    err_will = abs(will_final - exact_final)
    
    print("=========================================================")
    print("  FINANCIAL SDE BENCHMARK: GEOMETRIC BROWNIAN MOTION")
    print("=========================================================")
    print(f"True Final Stock Price:   ${exact_final:.4f}")
    print("---------------------------------------------------------")
    print(f"Euler-Maruyama Price:     ${em_final:.4f}  (Error: ${err_em:.4f})")
    print(f"Stochastic Williams:      ${will_final:.4f}  (Error: ${err_will:.4f})")
    print("=========================================================")
    
    if err_will < err_em:
        print("RESULT: SUCCESS!")
        print("The Williams Formula successfully handled the random market noise.")
        print("It provided a more accurate drift calculation than standard Euler-Maruyama.")
    else:
        print("RESULT: The standard noise broke the high-order advantage.")
        print("To beat Euler-Maruyama, the Williams Formula must be re-derived")
        print("using Ito Calculus to properly handle the stochastic term.")

if __name__ == '__main__':
    run_sde_benchmark()
