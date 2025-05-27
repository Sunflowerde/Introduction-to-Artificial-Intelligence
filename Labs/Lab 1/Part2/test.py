class MCTSAgent(MultiAgentSearchAgent):

    def getAction(self, gameState, mcts_time_limit=10):

        class Node:

            def __init__(self, data):
                self.north = None                   # 选择当前action为“north”对应的节点, <class 'Node'>
                self.east = None                    # 选择当前action为“east”对应的节点, <class 'Node'>
                self.west = None                    # 选择当前action为“west”对应的节点, <class 'Node'>
                self.south = None                   # 选择当前action为“south”对应的节点, <class 'Node'>
                self.stop = None                    # 选择当前action为“stop”对应的节点, <class 'Node'>
                self.parent = None                  # 父节点, <class 'Node'>
                self.statevalue = data[0]           # 该节点对应的游戏状态, <class 'GameState' (defined in pacman.py)>
                self.numerator = data[1]            # 该节点的分数
                self.denominator = data[2]          # 该节点的访问次数

        def Selection(cgs, cgstree):
            '''
                cgs: current game state, <class 'GameState' (defined in pacman.py)>
                cgstree: current game state tree, <class 'Node'>
                
                YOUR CORE HERE (~30 lines or fewer)
                1. You have to find a node that is not completely expanded (e.g., node.north is None)
                2. When you find the node, return its corresponding game state and the node itself.
                3. You should use best_UCT() to find the best child of a node each time.

            '''
            
            def uncompletelyExpanded(node):
                return (node.east is None and "East" in legalActions) or (node.south is None and "South" in legalActions) or (node.west is None and "West" in legalActions) or (node.north is None and "North" in legalActions)
            legalActions = cgs.getLegalActions(0)

            if len(legalActions) == 0:
                return (cgs, cgstree)
            
            if not uncompletelyExpanded(cgstree):
                successor = []

                if "East" in legalActions:
                    successor.append((cgstree.east, "East"))
                elif "South" in legalActions:
                    successor.append((cgstree.south, "South"))
                elif "West" in legalActions:
                    successor.append((cgstree.west, "West"))
                elif "North" in legalActions:
                    successor.append((cgstree.north, "North"))
                else:
                    successor.append((cgstree.stop, "Stop"))

                bestState, nextAction = best_UCT(successor)

                if nextAction == "East":
                    nextNode = cgstree.east
                elif nextAction == "South":
                    nextNode = cgstree.south
                elif nextAction == "West":
                    nextNode = cgstree.west
                elif nextAction == "North":
                    nextNode = cgstree.north
                else:
                    nextNode = cgstree.stop
                
                return Selection(bestState, nextNode)
            
            return (cgs, cgstree)

        def Expansion(cgstree):
            legal_actions = cgstree.statevalue.getLegalActions(0)
            '''
                YOUR CORE HERE (~20 lines or fewer)
                1. You should expand the current game state tree node by adding all of its children.
                2. You should use Node() to create a new node for each child.
                3. You can traverse the legal_actions to find all the children of the current game state tree node.
            '''

            def add(parent, child, action):
                if action == "East":
                    parent.east = child
                elif action == "South":
                    parent.south = child
                elif action == "West":
                    parent.west = child
                elif action == "North":
                    parent.north = child
                elif action == "Stop":
                    parent.stop = child
                else:
                    return None
            
            for action in legal_actions:
                nextState = cgstree.statevalue.generateSuccessor(0, action)
                child = Node([nextState, 0, 1])

                child.parent = cgstree
                add(cgstree, child, action)
            
            
        def Simulation(cgs, cgstree):
            '''
                This implementation is different from the one taught during the lecture.
                All the nodes during a simulation trajectory are expanded.
                We choose to more quickly expand our game tree (and hence pay more memory) to get a faster MCTS improvement in return.
            '''
            simulation_score = 0
            while cgstree.statevalue.isWin() is False and cgstree.statevalue.isLose() is False:
                cgs, cgstree = Selection(cgs, cgstree)
                Expansion(cgstree)
            '''
                YOUR CORE HERE (~4 lines)
                You should modify the simulation_score of the current game state.
            '''
            if cgs.isWin():
                simulation_score = 1
            elif cgs.isLose():
                simulation_score = 0

            return simulation_score, cgstree

        def Backpropagation(cgstree, simulation_score):
            while cgstree.parent is not None:
                '''
                    YOUR CORE HERE (~3 lines)
                    You should recursively update the numerator and denominator of the game states until you reaches the root of the tree.
                '''
                cgstree.numerator += simulation_score
                cgstree.denominator += 1
                cgstree = cgstree.parent
            return cgstree