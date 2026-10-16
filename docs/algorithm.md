# Certified spatial reaggregation

## Data model

Two agencies may release totals for different partitions of the same region.
Each nonempty geometric overlap becomes an edge in a bipartite network. Its
unknown nonnegative value is an overlap-cell allocation. The released totals
form the row and column marginals of that allocation.

An allocation is admissible when it is nonnegative, uses only retained overlap
edges, and reproduces every released marginal. A spatial query assigns a weight
to each overlap edge and sums the weighted allocation.

## Endpoint computation

The workflow minimises and maximises the query over the same admissible set.
These two optimisation problems return the sharp lower and upper endpoints.
They also return concrete allocations attaining both endpoints. If either
problem is infeasible, a violated neighbour cut identifies incompatible
marginals or an incomplete overlap support.

## Exactness

Balanced perturbations of a feasible allocation lie in the null space of the
unsigned incidence matrix. The workflow constructs the network cycle space and
tests whether a query is orthogonal to every feasible cycle direction. When it
is, every admissible allocation has the same query value. Numerical endpoint
equality is checked independently against this structural criterion.

## Verification

The evaluation combines several independent checks:

- primal feasibility and objective replay for both endpoint allocations;
- dual certificate replay from solver marginals;
- integer min-cost flow on integral instances;
- complete enumeration on small networks;
- cycle bases built by combinatorial and numerical procedures;
- deliberate certificate mutations that must be rejected;
- support-cutoff and edge-deletion perturbations.

Each check writes a machine-readable CSV file. `results/oracle_audit.csv`
collects the check counts and failure counts in one place.
