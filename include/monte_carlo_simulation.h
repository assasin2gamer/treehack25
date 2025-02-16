#ifndef MONTE_CARLO_SIMULATION_H
#define MONTE_CARLO_SIMULATION_H

#include <vector>

// Offloads Monte Carlo motion generation (weighted jump & Brownian motion) to CUDA.
// Parameters:
//   startPrice    - initial stock price
//   nPaths        - number of simulation paths
//   nSteps        - number of time steps per simulation
//   dt            - time step (e.g. 1/252 for daily steps)
//   mu            - drift
//   sigma         - volatility
//   jumpProb      - probability of a jump
//   jumpMagnitude - magnitude factor for the jump
// Returns a vector of simulated price paths (each path is a vector of doubles).
std::vector<std::vector<double> > simulateMonteCarloPaths(double startPrice, int nPaths, int nSteps,
                                                          double dt, double mu, double sigma,
                                                          double jumpProb, double jumpMagnitude);

#endif // MONTE_CARLO_SIMULATION_H
