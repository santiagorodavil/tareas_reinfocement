import numpy as np
import random
import operator

class EscalerasSerpientes:
    def __init__(self,goals,fails,live_reward=0.0):
        
        ## Inicializa los atributos de la clase
        self.goals = goals
        self.fails = fails
        self.snakes = {73:1, 46:5, 55:7, 48:9, 52:11, 59:17, 83:19, 44:22, 95:24, 98:28, 69:33, 64:36,92:51}
        self.stairs =  {8:26, 21:82, 43:77, 50:91, 54:93, 62:96, 66:87, 80:100}
        self.live_reward = live_reward

        # S
        self.states = []
        # A(S)
        self.allowed_actions = {}

        # Generador de estados (tablero) y a(s)
        for n in range(100):
            state=n+1
            self.states.append(state)
            self.allowed_actions[state] = ['Ad','At']
        
        # Inicializa el valor de cada estado
        self.state_values = self.init_values()
        
        # Q(s,a)
        self.Q=self.init_Q(0,0)
        
        # C(s,a)
        self.C=self.init_Q(0,0)
        
        # Inicializa la política
        self.policy = dict.fromkeys(self.states, 'Ad')

        # Acción real (aleatoria) del agente dada una acción determinística
        self.real_actions = {'Ad': [1,2,3,4,5,6],
                             'At': [-1,-2,-3,-4,-5,-6]}

        # Probabilidades de las acciones reales
        self.action_probabilities = [1/6,1/6,1/6,1/6,1/6,1/6]

        # Suma cumulativa de probabilidades
        self.action_probs_cum = np.cumsum(self.action_probabilities)

    # Método de inicialización de estados
    def init_values(self):
        state_values = {}
        for state in self.states:
            state_values[state] = 0.0
        return state_values
    
    # Método de inicialización de par estado accion
    def init_Q(self,Ad,At):
        Q = {}
        for state in self.states:
            Q[state] = {'Ad': Ad,'At': At}
        return Q

    # Método para obtener las acciones permitidas en un estado
    def get_allowed_actions(self, state):
        return self.allowed_actions[state]

    # Método para dar pasos
    def step(self, state, action, random=False):
        if(state in self.goals):
            return "Azul", +1.0, None,None, True
        elif(state in self.fails):
            return "Rojo", -1.0,None, None, True
        else:
            if random:
                assert action in self.get_allowed_actions(state)
                rand = np.random.uniform()
                if rand<=self.action_probs_cum[0]:
                    real_action = self.real_actions[action][0]
                elif rand>self.action_probs_cum[0] and rand<=self.action_probs_cum[1]:
                    real_action = self.real_actions[action][1]
                elif rand>self.action_probs_cum[1] and rand<=self.action_probs_cum[2]:
                    real_action = self.real_actions[action][2]
                elif rand>self.action_probs_cum[2] and rand<=self.action_probs_cum[3]:
                    real_action = self.real_actions[action][3]
                elif rand>self.action_probs_cum[3] and rand<=self.action_probs_cum[4]:
                    real_action = self.real_actions[action][4]
                else:
                    real_action = self.real_actions[action][5]
            else:
                real_action = action
            
            # Actualiza el estado resultante teniendo en cuenta el rebote
            if state+real_action<self.states[0] :
                # (Movidas restantes al llegar al estado 0) + (estado 0)
                state_=(abs(real_action)-(state-self.states[0]))+self.states[0]
            elif state+real_action>self.states[-1]:
                # (estado 100)-(Movidas restantes al llegar al estado 100) 
                state_=self.states[-1]-(abs(real_action)-(self.states[-1]-state))
            else:
                state_=state+real_action

            # Actualiza el estado resultante teniendo en cuenta las escaleras y serpientes
            if state_ in self.snakes:
                state_=self.snakes[state_]
            if state_ in self.stairs:
                state_=self.stairs[state_]

            # Evalúa recompensas de la transición
            if state_ not in self.states:
                # Esto no debería pasar
                print(f"Se salió del tablero. Intento llegar a: {state_}")
                reward = self.live_reward
                state_ = state
                done = False
            else:
                reward = self.live_reward
                done = False

            return state_, reward, action, real_action, done
    
    # Métodos auxiliares
    def key_max(self, d):
        return max(d.items(), key=operator.itemgetter(1))

    def max_val(self, d):
        return max(d.items(), key=operator.itemgetter(1))[1]

