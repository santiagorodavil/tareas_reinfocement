import numpy as np
class Game_hw2:
    snakes = [(73,1), (46,5), (55,7), (48,9), (52,11), (59,17), (83,19), (44,22), (95,24), (98,28), (69,33), (64,36),(92,51)]
    stairs = [(8,26), (21,82), (43,77), (50,91), (54,93), (62,96), (66,87), (80,100)]
    def __init__(self, goals, fails, snakes = snakes, stairs = stairs):
        self.goals = goals
        self.fails = fails
        self.snakes = snakes
        self.stairs = stairs
        self.states = []
        
        
        self.probabilities = {}
        
        #Crear el tablero
        value = 0
        for n in range(10):
            row = []
            for i in range(10):
                value+=1        
                row.append(value)
            self.states.append(row)
            
            
        # Diccionario de probabilidades adelante/atrás
        # Estructura-> casilla actual: probabilidad, casilla de llegada (segun el valor del dado), Recompensa.
        # Como se lanza 1 dado, la probabilidad siempre es de 1/6 para cualquiera de los resultados.   
        p_caida = 1/6
        reward = 0
        self.goal_rew = 1
        self.fail_rew = -1
        for row in self.states:
            for index in row:  
                mov_prob = {'Ad':[],'At':[]}
                for dice in range(1,7):
                    # casilla de llegada si elijo adelante(ad)/atras(at)
                    cas_lleg_ad = index+dice
                    cas_lleg_at = index-dice

                    # Si llega a salirse del borde, se hace rebotar
                    if cas_lleg_ad >=101:
                        cas_lleg_ad = 100 - (index+dice -100)

                    elif cas_lleg_at <= 0:
                        cas_lleg_at = abs(index-dice)+2

                    #asignar casilla llegada en caso de que sea sepriente
                    for snake in snakes:
                        if cas_lleg_ad == max(snake):
                            #print(cas_lleg_ad)
                            cas_lleg_ad = min(snake)
                        elif cas_lleg_at == max(snake):
                            cas_lleg_at = min(snake)

                    #asignar casilla llegada en caso de que sea escalera
                    for stair in stairs:
                        if cas_lleg_ad == min(stair):
                            cas_lleg_ad = max(stair)
                        elif cas_lleg_at == min(stair):
                            cas_lleg_at = max(stair)

                    # Asigna recompensa a la respectiva casilla de llegada (normal, goal o fail)
                    if cas_lleg_ad in goals: current_rew_ad = self.goal_rew
                    elif cas_lleg_ad in fails: current_rew_ad = self.fail_rew
                    else: current_rew_ad = reward

                    if cas_lleg_at in goals: current_rew_at = self.goal_rew
                    elif cas_lleg_at in fails: current_rew_at = self.fail_rew
                    else: current_rew_at = reward

                    # Se le asigna a 
                    mov_prob['Ad'].append([p_caida, cas_lleg_ad, current_rew_ad])
                    mov_prob['At'].append([p_caida, cas_lleg_at, current_rew_at])
                self.probabilities[index] = mov_prob
            
        self.state_values = self.init_values()
        self.policy = self.init_policy()
    
    def init_values(self):
        state_values = {}
        for n in range(100):
            state_values[n+1] = 0
        return state_values
    
    def init_policy(self):
        policy = {}
        for n in range(100):
            policy[n+1] = 'Ad'
        return policy
        
                
    def step(self, state, action, random=False, dice=0):
        posible_actions = self.probabilities[state]
        # valor de arrojar el dado
        if dice == 0:
            dice = np.random.randint(1, 7)
        
        # si la politica es aleatoria, se le asigna una accion aleatoria
        if random:
            rand_act = np.random.uniform()
            if rand_act > 0.5:
                real_action = 'Ad'
            else:
                real_action = 'At'
        else:
            real_action = action
        
        # Asigna el valor del estado siguiente dependiendo de la accion dada y del valor del dado
        if real_action == 'Ad':
            next_state = posible_actions[real_action][dice-1]
        if real_action == 'At':
            next_state = posible_actions[real_action][dice-1]
        
        # Evalua si next_state es un estado terminal
        if (next_state[1] in self.goals) or (next_state[1] in self.fails):
            done = True
        else:
            done  = False
        
        reward = next_state[2]
        
        return next_state[1], reward, real_action, done, dice
        
                