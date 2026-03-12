import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn import linear_model


class SLR_slope_simulator:
    def __init__(self, beta_0, beta_1, x, sigma, seed):
        # save the model settings used to generate data
        self.beta_0 = float(beta_0)
        self.beta_1 = float(beta_1)
        self.sigma = float(sigma)

        # save the fixed x values and sample size
        self.x = np.asarray(x, dtype=float)
        self.n = len(self.x)

        # create a seeded random number generator
        self.rng = default_rng(seed)

        # this will later be replaced by an array of simulated slope estimates
        self.slopes = []

    def generate_data(self):
        # generate one simulated dataset from the SLR model
        eps = self.rng.normal(loc=0.0, scale=self.sigma, size=self.n)
        y = self.beta_0 + self.beta_1 * self.x + eps
        return self.x, y

    def fit_slope(self, x, y):
        # fit the SLR model and return only the estimated slope
        reg = linear_model.LinearRegression()
        fit = reg.fit(np.asarray(x, dtype=float).reshape(-1, 1),
                      np.asarray(y, dtype=float))
        return float(fit.coef_[0])

    def run_simulations(self, num_sims):
         
        num_sims = int(num_sims)
        beta_array = np.zeros(num_sims)

        for i in range(num_sims):
            # repeatedly generate data
            x_sim, y_sim = self.generate_data()
            #fit the line and save the slope estimate
            beta_array[i] = self.fit_slope(x_sim, y_sim)
        
        self.slopes = beta_array

    def plot_sampling_distribution(self):
        # check that run_simulations() has been called before plotting
        if len(self.slopes) == 0:
            print("run_simulations() must be called first.")
            return None

        plt.hist(self.slopes)
        plt.title("Sampling distribution of simulated SLR slope")
        plt.xlabel("Estimated slope")
        plt.ylabel("Count")
        plt.show()

    def find_prob(self, value, sided):
        # use the simulated slopes to find a probability
        if len(self.slopes) == 0:
            print("run_simulations() must be called first.")
            return None

        value = float(value)
        sided = str(sided).lower()

        if sided == "above":
            return float(np.mean(self.slopes > value))

        if sided == "below":
            return float(np.mean(self.slopes < value))

        if sided == "two-sided":
            med = float(np.median(self.slopes))

            if value >= med:
                prob = 2 * np.mean(self.slopes > value)
            else:
                prob = 2 * np.mean(self.slopes < value)

            return float(min(1.0, prob))

        raise ValueError("sided must be one of: 'above', 'below', or 'two-sided'.")






if __name__ == "__main__":
    # create the x values 
    x = np.array(list(np.linspace(start=0, stop=10, num=11)) * 3)

    # create the simulator object 
    sim = SLR_slope_simulator(
        beta_0=12,
        beta_1=2,
        x=x,
        sigma=1,
        seed=10
    )

    # call the plot method before simulations
    # this checks whether the slopes attribute has been filled yet
    sim.plot_sampling_distribution()

    # then run the simulations and produce the requested results
    sim.run_simulations(10000)
    # plot the simulated sampling distribution
    # this histogram is using the saved slope estimates as an approximation to the sampling distribution
    sim.plot_sampling_distribution()

    # approximate the two-sided probability for 2.1
    # this uses the simulated slopes and the class's two-sided rule to estimate a probability
    p_two = sim.find_prob(2.1, sided="two-sided")
    print("Approximate two-sided probability for 2.1:", p_two)

    # print the simulated slopes attribute
    print(sim.slopes)
