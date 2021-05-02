# Natural Disasters, Climate Change, and Sovereign Risk

# Overview


# Data Preparation
In his paper Mallucci uses data sources from the World Bank regarding macroeconomics features in countries in the Caribbean, we will need to construct routines to fetch the data and wire in the model, notably:

- [ ] Government bond duration
  - Average maturity (TBD)
- [x] Hurricane Frequency
  - [https://coast.noaa.gov/hurricanes/#map=4/32/-80](https://coast.noaa.gov/hurricanes/#map=4/32/-80)
  -[https://bit.ly/3xGETb7](https://bit.ly/3xGETb7)
  
We can then use the GDP/GNI World Bank through the [wbdata](https://github.com/OliverSherouse/wbdata) API to estimate:
```math
log(y') = \rho log(y) -\ksi h + \epsilon^y
```
- [x] Get GDP data from the World Bank
  - [GDP](](../src/sovereign_risk_disaster/data/data_wb.py))
- [ ] Endowment autocorrelation
  - point estimate of the lagged GDP
- [ ] Endowment st. dev
  - standard deviation of the regressions errors
- [ ] Hurricane mean loss
  - regression coefficient for the dummy variable of hurricane hits
- [ ] Hurricane loss st. dev
  - standard deviation of the coefficient of the dummy variable

  
- [ ] Discount Factor and output cost
   - Method of moments: average spread between US T-bills and local gov bond nd average debt-to-GDP ratio
  

# Algorithm 

## Definitions

## Example 

## Computation Flow

In his Paper Mallucci describes an extended sovereign default model on the same basis as [Arellano2008](#references) , including disaster risk considerations

For the baseline economy:
1. [x] Discretize income processes y and determine the transition matrix $`Y′|Y`$ using the quadrature method for the normal distribution described in Tauchen and Hussey (1991) 
   - [Tauchen](../src/sovereign_risk_disaster/markov/tauchen.py)
2. [ ] Set up the grid of states $`\Omega = {y\times h \times b}`$ and choices $`{b′}`$
3. [ ] Guess an arbitrary price $`q`$ for government bonds
4. [ ] Guess initial values for the vale functions $`V^{nd}`$ and $`V^d`$
5. [ ] Compute utilities and continuation values on each point of the grid
6. [ ] Iterate value functions till convergence
7. [ ] Update the price of government debt
8. [ ] Repeat steps (4)-(7) until the price of government debt has converged.

We will then replicate the model with the disaster clause which extend the above model.

# References
- Mallucci, Enrico, Natural Disasters, Climate Change, and Sovereign Risk (July, 2020). FRB International Finance Discussion Paper No. 1291, 
     - Available at SSRN: [https://ssrn.com/abstract=3648271](https://ssrn.com/abstract=3648271)
- Arellano, Cristina, “Default Risk and Income Fluctuations in Emerging Economies,” American Economic Review, 2008, 98 (3), 690–712.  
