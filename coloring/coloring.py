from pulp import *
import igraph


def coloring(graph, wmax=None, preserve=False, weights='weight'
             , color_lbl='colour', relevant_lbl='relevant'):
    """

    A method for vertex coloring using the minimum number of colors with
    maximum number of vertices per color restriction

    :param graph: A graph with vertices that represents tasks.
                  Two adjacent vertices can't have the same color.
    :param wmax: Float, maximum weight per color.
    :param preserve: Preserve pre-existing colors.
    :param weights: weight of each task.
    :param color_lbl: A label for graph.vs attribute, in order to store
                      vertex color.
    :param relevant_lbl: A label for graph.vs attribute, True if vertex
                         should be included in the problem formulation.
    :return: 0 (No solution found), 1 (Optimal)
    """

    epsilon = 0.5

    assert isinstance(graph, igraph.Graph)

    try:
        graph.vs[color_lbl]
    except KeyError:
        graph.vs[color_lbl] = [None] * len(graph.vs)

    try:
        graph.vs[relevant_lbl]
    except KeyError:
        graph.vs[relevant_lbl] = [True] * len(graph.vs)

    try:
        graph.vs[weights]
    except KeyError:
        graph.vs[weights] = [1.0] * len(graph.vs)

    if not wmax:
        wmax = sum(graph.vs[weights])

    if not preserve:
        graph.vs[color_lbl] = [None] * len(graph.vs)

    p = _colouring_lpproblem(graph, wmax, weights, color_lbl, relevant_lbl)
    p.solve()
    if p.status != 1:
        return 0

    for var in p.variables():
        if var.varValue > epsilon and var.name[:4] == "Node":
            s = var.name
            s = s[s.find('_(') + 2:-1]
            s = s.split(",_")
            s = [int(si) for si in s]
            graph.vs[s[0]][color_lbl] = s[1]
    return 1


def _colouring_lpproblem(graph, wmax, weights='weight', color_lbl='colour'
                         , relevant_lbl='relevant', instance_name="NN"):
    """

    :param graph: A graph with vertices that represents tasks.
                  Two adjacent vertices can't have the same color.
    :param cmax: Integer, maximum number of tasks per color.
    :param color_lbl: A label for graph.vs attribute, in order to store
                      vertex color.
    :param relevant_lbl: A label for graph.vs attribute, True if vertex
                         should be included in the problem formulation.
    :param instance_name: A name for LpProblem
    :return: A pulp.LpProblem instance with the problem formulation
    """
    assert isinstance(graph, igraph.Graph)

    relevants = [i for i in range(len(graph.vs)) if graph.vs[i][relevant_lbl]]

    big_number = len(relevants) * sum(range(len(graph.vs))) * 2

    prob = LpProblem('Constrained Colouring Problem: %s' % instance_name
                     , LpMinimize)

    # Variables: xc one iff color c is used, 0 otherwise.
    xc_combs = [c for c in range(len(graph.vs))]
    xc = LpVariable.dicts('Colour variable x(c)', xc_combs, lowBound=0
                          , upBound=1, cat=LpInteger)

    # Variable: xic one iff vertex i uses color c, 0 otherwise.
    xic_combs = []
    for i in relevants:
        for c in range(len(graph.vs)):
            xic_combs.append((i, c))
    xic = LpVariable.dicts('Node Colour variable x(i,c)', xic_combs
                           , lowBound=0, upBound=1, cat=LpInteger)

    # Objective: Minimize sum of all colours

    obj = ""
    for c in range(len(graph.vs)):
        obj += " + %f * xc[%d]" % (big_number, c)

    for i in relevants:
        for c in range(len(graph.vs)):
            obj += " + %f * xic[(%d,%d)]" % (c, i, c)
    prob += eval(obj)

    # Constraint: At least one color per vertex and at most one color
    #             per vertex
    for i in relevants:
        constr = ""
        for c in range(len(graph.vs)):
            constr += " + xic[(%d,%d)]" % (i, c)
        constr += " == 1"
        prob += eval(constr)

    # Constraint: Two adjacent vertices can't have the same color
    for e in graph.es:
        if e.source in relevants and e.target in relevants:
            for c in range(len(graph.vs)):
                prob += xic[(e.source, c)] + xic[(e.target, c)] <= 1

    # Constraint: Maximum number of tasks per color
    for c in range(len(graph.vs)):
        constr = ""
        for i in relevants:
            constr += " + %f * xic[(%d,%d)]" % (graph.vs[i][weights], i, c)
        constr += " <= %f" % wmax
        prob += eval(constr)

    # Constraint: One color is used iff at least one vertex uses it
    for i in relevants:
        for c in range(len(graph.vs)):
            prob += xc[c] - xic[(i, c)] >= 0

    # Constraint: Pre-existing colors persistence
    for i, v in enumerate(graph.vs):
        if v[color_lbl] is not None and i in relevants:
            prob += xic[(i, v[color_lbl])] == 1
    return prob
