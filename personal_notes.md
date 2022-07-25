## How do you define a Markov chain?

There are 3 items involved:
- State space, S [Finite/countable set of states]
- Initial distribution:
    Probability distribution of the Markov chain at time =0.
    $$\sum_{i \epsilon S} \pi_0(i)) = 1$$
- Probability transition rule

## Value Iteration

- It first initializes all values $V_curr$ at time=0 arbitrarily.
- In the $curr$ iteration, it makes a full sweep, i.e., iterates over all states, computing a new approximation $V_new$ for the state values using the successor values from the previous iteration $(V_{curr})$ as

### Algorithm:
- Start with $V_{curr}(s) = 0$ for all $s$
- error = $\infty$
- While error > $\epsilon$
    - For each state $s$
        - $V_{new}(s) = \max_{a\in A} R(s,a) + \beta \sum_{s'\in S} T(s,a,s') V_{curr}(s')$
        - $\pi_{curr}(s) = \arg\max_{a\in A} R(s,a) + \beta \sum_{s'\in S} T(s,a,s') V_{curr}(s')$
    - error = $\max_s |V_{new}(s)-V_{curr}(s)|$   ;; could do this incrementally
    - $V_{curr} = V_{new}$


