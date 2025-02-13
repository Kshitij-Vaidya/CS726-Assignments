# CS726

## Programming Assignment 1
cs726_pass_1.ipynb has Junction tree with separator sets for message passing constructed. 

Useful methods for message passing: 

self.maximal_cliques = list of cliques. Each clique = list of nodes in that clique, eg C1 = [0,1]

self.junction_tree = list of tuple, each tuple of the form (clique i, clique j, separator_set_i->j, separator_set_j->i) (all elements in this tuple are lists of nodes)

self.assigned_clique_potentials = dictionary[key = tuple containing clique nodes eg (0,1)] -> value = list containing potentials in order like sample
