import numpy as np
import pygame
from time import sleep
import random
import operator
from colorama import Fore
#from learning_curve import LearningCurve

random.seed(0)

# RENDER
CELL_SIZE = 150
AGENT_RADIUS = 50
MARGIN = 20
MARGIN_GOAL = 6
LINES_WIDTH = 3
REWARD_TEXT_SIZE = 40
VALUE_TEXT_SIZE = 35
Q_VALUE_TEXT_SIZE = 28

BLACK = np.array([  0,   0,   0])
GRAY =  np.array([100, 100, 100])
WHITE = np.array([255, 255, 255])
BLUE =  np.array([  0,   0, 200])
GREEN = np.array([  0, 150,   0])
RED =   np.array([150,   0,   0])
PINK =  np.array([255, 153, 177])

class GridWorld:
    def __init__(self, rows, cols, walls=[], pits=[], goals=[], live_reward=0.0):
        self.rows = rows
        self.cols = cols
        self.walls = walls
        self.goals = goals
        self.pits = pits
        self.live_reward = live_reward

        self.states = []
        self.allowed_actions = {}

        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                #if (state not in self.walls) and (state not in self.pits) and (state not in self.goals):
                if (state not in self.walls):
                    self.states.append(state)
                self.allowed_actions[state] = ['N','S','E','W']

        self.state_values = self.init_values()
        self.state_q_values = self.init_qvalues()
        self.policy = dict.fromkeys(self.states, 'N')

        self.real_actions = {'N': ['N', 'E', 'W'],
                             'S': ['S', 'E', 'W'],
                             'E': ['E', 'N', 'S'],
                             'W': ['W', 'N', 'S']}

        self.action_probabilities = [0.8, 0.1, 0.1]
        self.action_probs_cum = np.cumsum(self.action_probabilities)

        self.r_init_render()

    def init_values(self):
        state_values = {}
        for state in self.states:
            state_values[state] = 0.0
        return state_values

    def init_qvalues(self):
        state_q_values = {}
        for state in self.states:
            state_q_values[state] = dict.fromkeys(self.allowed_actions[state], 0.0)
        return state_q_values

#     def init_values(self):
#         state_values = {}
#         state_q_values = {}
#         for state in self.states:
#             state_values[state] = 0.0
#             state_q_values[state] = dict.fromkeys(self.allowed_actions[state], 0.0)
#         return state_values, state_q_values


    def updateGrid(self):
        self.r_draw_values()

    def get_allowed_actions(self, state):
        return self.allowed_actions[state]

    def step(self, state, action, random=False):
        if(state in self.goals):
            return "Terminal Diamante", +1.0, None, True
        elif(state in self.pits):
            return "Terminal Bomba", -1.0, None, True
        else:
            assert action in self.get_allowed_actions(state)
            if random:
                rand = np.random.uniform()
                if rand<=self.action_probs_cum[0]:
                    real_action = self.real_actions[action][0]
                elif rand>self.action_probs_cum[0] and rand<=self.action_probs_cum[1]:
                    real_action = self.real_actions[action][1]
                else:
                    real_action = self.real_actions[action][2]
            else:
                real_action = action

            if real_action=='N':
                state_ = (state[0]-1, state[1])
            if real_action=='S':
                state_ = (state[0]+1, state[1])
            if real_action=='W':
                state_ = (state[0], state[1]-1)
            if real_action=='E':
                state_ = (state[0], state[1]+1)

            if state in self.pits:
                reward = -1.0
                done = True
            elif state in self.goals:
                reward = 1.0
                done = True
            elif state_ not in self.states:
                reward = self.live_reward
                state_ = state
                done = False
            else:
                reward = self.live_reward
                done = False

            return state_, reward, real_action, done

    def wait_esc_key(self):
        clock = pygame.time.Clock()
        while(True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
            clock.tick(60)

    def wait_space_key(self):
        clock = pygame.time.Clock()
        while(True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return pygame.K_ESCAPE
                if event.type == pygame.KEYDOWN:
                    return event.key
                    # if event.key == pygame.K_ESCAPE:
                    #     return -1
                    # if event.key == pygame.K_SPACE:
                    #     return 0
            clock.tick(60)

    def tick_key(self, fps):
        clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return pygame.K_ESCAPE
            if event.type == pygame.KEYDOWN:
                return event.key
        clock.tick(fps)

    # def key_max(self, d):
    #     max_val = np.finfo(float).min
    #     max_key = None
    #     for key in d:
    #         if d[key]>max_val:
    #            max_val = d[key]
    #            max_key = key
    #     print(max_val, max_key)
    #     return max_val, max_key

    def key_max(self, d):
        '''key, val'''
        return max(d.items(), key=operator.itemgetter(1))

    def max_val(self, d):
        return max(d.items(), key=operator.itemgetter(1))[1]


    # Execution modes

    def solve_dynamic_programming(self, gamma=1.0, horizon=10, init_state=(2,0)):
        escape = False
        for i in range(horizon):
            self.r_draw_background()
            self.r_draw_values(show_action=False)
            self.r_draw_iteration(i+1, flagP='dynamic', flagT=False)
            pygame.display.flip()
            key = self.wait_space_key()
            if key==pygame.K_ESCAPE:
                escape = True
                break
            elif key==pygame.K_SPACE:
                try:
                    self.update_values(gamma)
                except Exception as e:
                    escape = True
                    print(Fore.RED+"Error en la función ´update_values´")
                    print(e)
                    break
        if not escape:
            self.wait_esc_key()
        pygame.quit()

    def solve_value_iteration(self, gamma=1.0, horizon=10, init_state=(2,0)):
        escape = False
        for i in range(horizon):
            self.r_draw_background()
            self.r_draw_values(show_action=False)
            self.r_draw_iteration(i+1, flagP='value', flagT=False)
            pygame.display.flip()
            key = self.wait_space_key()
            if key==pygame.K_ESCAPE:
                escape = True
                break
            elif key==pygame.K_SPACE:
                try:
                    self.value_iteration(gamma)
                except Exception as e:
                    escape = True
                    print(Fore.RED+"Error en la función ´value_iteration´")
                    print(e)
                    break
        self.r_draw_iteration(i+1, flagP='value', flagT=True)
        while(key!=pygame.K_RETURN):
            #print(f'pressed={key}, t={pygame.K_RETURN}')
            key = self.wait_space_key()
            if key==pygame.K_ESCAPE:
                escape = True
                break
        if not escape:
            self.follow_policy(init_state=init_state, gamma=gamma)
        pygame.quit()

    def solve_policy_iteration(self, gamma=1.0, horizon=10, init_state=(2,0)):
        policy_stable = False
        flag_q = False
        escape = False
        while(not policy_stable):
            for i in range(horizon):
                self.r_draw_background()
                if not flag_q:
                    self.r_draw_values()
                else:
                    self.r_draw_q_values()
                self.r_draw_iteration(i+1, flagP='policy', flagT=False)
                pygame.display.flip()
                key = self.wait_space_key()
                if key==pygame.K_ESCAPE:
                    escape = True
                    break
                elif key==pygame.K_RSHIFT or key==pygame.K_LSHIFT:
                    flag_q = not flag_q
                elif key==pygame.K_SPACE:
                    try:
                        self.policy_evaluation(gamma)
                    except Exception as e:
                        escape = True
                        print(Fore.RED+"Error en la función ´policy_evaluation´")
                        print(e)
                        break
            try:
                policy_stable = self.policy_improvement()
            except Exception as e:
                escape = True
                print(Fore.RED+"Error en la función ´policy_improvement´")
                print(e)
                break
        if not escape:
            self.r_draw_iteration(i+1, flagP='policy', flagT=True)
            while(key!=pygame.K_RETURN):
                key = self.wait_space_key()
                if key==pygame.K_ESCAPE:
                    escape = True
                    break
            if not escape:
                self.follow_policy(init_state=init_state, gamma=gamma)
        pygame.quit()



    def solve_bellman(self, gamma=1.0):
        ''' Value iteration
        q: change view (state values or q values)
        space: iteration
        escape: exit
        t: test current policy'''

        self.state_values, self.state_q_values = self.init_values()
        i = 0
        flag_q = False
        while(True):
            self.r_draw_background()
            if not flag_q:
                self.r_draw_values()
            else:
                self.r_draw_q_values()
            pygame.display.flip()
            key = self.wait_space_key()

            if key==pygame.K_ESCAPE:
                break

            elif key==pygame.K_q:
                flag_q = not flag_q

            elif key==pygame.K_SPACE:
                value_iteration(self, H)
#                 state_values_1, state_q_values_1 = self.init_values()
#                 for state in self.states:
#                     for action in self.get_allowed_actions(state):
#                         q = 0.0
#                         for j, raction in enumerate(self.real_actions[action]):
#                             state_, rew, _, _ = self.step(state, raction, False)
#                             if state_ not in self.states:
#                                 q += self.action_probabilities[j]*rew
#                             else:
#                                 q += self.action_probabilities[j]*( rew + gamma*self.state_values[state_] )
#                         state_q_values_1[state][action] = q
#                     self.policy[state], state_values_1[state] = self.key_max(state_q_values_1[state])
#                 self.state_values = state_values_1
#                 self.state_q_values = state_q_values_1
                i += 1

            elif key==pygame.K_t:
                self.follow_policy(flag_q=flag_q, gamma=gamma)
                self.wait_esc_key()

    def q_learning( self, gamma=0.9, alpha=0.3, episodes=100, max_steps=50, fps=30, epsilon_0=-1.0,
                    plot=False):
        ''' Q learning
        q: change view (state values or q values)
        s: change speed (slow or fast)
        e: Explore or not
        '''

        if plot:
            l_curve = LearningCurve(min_y=-1.5, max_y=1.5)

        self.state_values, self.state_q_values = self.init_values()
        flag_q = False
        flag_fast = False
        flag_exit = False
        flag_explore = True

        episode=0
        while(True):
        # for episode in range(episodes):
            if flag_explore:
                if epsilon_0>=0.0:
                    epsilon = epsilon_0
                else:
                    epsilon = np.exp(-episode/(episodes/5))
            else:
                epsilon = 0.0
            state = random.choice(self.states)
            done = False
            explore=False
            action=''
            utility = 0.0
            reward = 0.0
            for step in range(max_steps):
            # while(True):
                self.r_draw_background()
                if not flag_q:
                    self.r_draw_values()
                else:
                    self.r_draw_q_values()
                self.r_draw_agent(state)
                self.r_draw_reward(reward, utility, done)
                # self.r_draw_rl_metrics(f'{episode+1}/{episodes}', epsilon, action, explore)
                self.r_draw_rl_metrics(episode+1, epsilon, action, explore)
                pygame.display.flip()
                if flag_fast:
                    key = self.tick_key(fps)
                else:
                    key = self.tick_key(1)
                if key==pygame.K_q:
                    flag_q = not flag_q
                elif key==pygame.K_s:
                    flag_fast = not flag_fast
                elif key==pygame.K_e:
                    flag_explore = not flag_explore
                elif key==pygame.K_ESCAPE:
                    flag_exit = True
                    break
                if done:
                    break
                if np.random.uniform()<epsilon:
                    explore = True
                    action = random.choice(self.allowed_actions[state])
                else:
                    explore = False
                    action = self.policy[state]
                new_state, reward, _, done = self.step(state, action)
                if done:
                    sample = reward
                else:
                    sample = reward + gamma*self.max_val(self.state_q_values[new_state])
                self.state_q_values[state][action] = (1-alpha)*self.state_q_values[state][action] + alpha*sample
                self.policy[state], self.state_values[state] = self.key_max(self.state_q_values[state])
                utility += (gamma**step)*reward
                state = new_state
            if plot:
                l_curve.add_sample(episode, utility)
            if flag_exit:
                break
            episode+=1



    def follow_policy(self, init_state=(-1,-1), flag_q=False, gamma=1.0):
        if init_state==(-1,-1):
            state = random.choice(self.states)
        else:
            state = init_state
        utility = 0.0
        reward = 0.0
        i = 0
        done = False
        while True:
            self.r_draw_background()
            if not flag_q:
                self.r_draw_values()
            else:
                self.r_draw_q_values()
            self.r_draw_agent(state)
            self.r_draw_reward(reward, utility, done)
            pygame.display.flip()
            sleep(0.3)
            if done:
                break
            action = self.policy[state]
            state, reward, _, done = self.step(state, action, random=True)
            utility += (gamma**i)*reward
            i += 1



    def start_manual_mode(self, init_state):
        self.curr_state = init_state
        assert self.curr_state in self.states
        assert self.curr_state not in self.goals
        assert self.curr_state not in self.pits
        clock = pygame.time.Clock()
        taken_action = True
        utility = 0.0
        done = False
        reward = 0.0
        while(True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_w:
                        self.curr_state, reward, _, done = self.step(self.curr_state, 'N')
                        taken_action = True
                    if event.key == pygame.K_s:
                        self.curr_state, reward, _, done = self.step(self.curr_state, 'S')
                        taken_action = True
                    if event.key == pygame.K_a:
                        self.curr_state, reward, _, done = self.step(self.curr_state, 'W')
                        taken_action = True
                    if event.key == pygame.K_d:
                        self.curr_state, reward, _, done = self.step(self.curr_state, 'E')
                        taken_action = True
            if taken_action:
                utility += reward
                print(f'reward:{reward}, utility:{utility}, done:{done}')
                self.r_draw_background()
                self.r_draw_agent(self.curr_state)
                self.r_draw_reward(reward, utility, done)
                if done:
                    break
            taken_action = False
            clock.tick(60)
        self.wait_esc_key()

    #RENDER
    def r_state_to_rect(self, state, goal=False):
        row = state[0]
        col = state[1]
        left = MARGIN+col*CELL_SIZE
        top = MARGIN+row*CELL_SIZE
        if not goal:
            return [left, top, CELL_SIZE, CELL_SIZE]
        else:
            return [left+MARGIN_GOAL, top+MARGIN_GOAL, CELL_SIZE-2*MARGIN_GOAL, CELL_SIZE-2*MARGIN_GOAL]

    def r_init_render(self):
        pygame.init()
        pygame.display.set_caption("Grid World - MDP")
        self.font_reward = pygame.font.SysFont('Arial', REWARD_TEXT_SIZE)
        self.font_value = pygame.font.SysFont('Arial', VALUE_TEXT_SIZE)
        self.font_q_value = pygame.font.SysFont('Arial', Q_VALUE_TEXT_SIZE)

        self.size_screen = [2*MARGIN+self.cols*CELL_SIZE, 2*MARGIN+self.rows*CELL_SIZE + 3*REWARD_TEXT_SIZE + MARGIN]
        self.screen = pygame.display.set_mode(self.size_screen)
        self.r_draw_background()

    def r_draw_background(self):
        self.screen.fill(BLACK)
        for state in self.states:
            pygame.draw.rect(self.screen, WHITE, self.r_state_to_rect(state), LINES_WIDTH)
        for goal in self.goals:
            pygame.draw.rect(self.screen, WHITE, self.r_state_to_rect(goal), LINES_WIDTH)
            pygame.draw.rect(self.screen, GREEN, self.r_state_to_rect(goal, True) )
            img = pygame.image.load('images/diamante.png')
            img.convert()
            self.screen.blit(img, self.r_state_to_rect(goal))
            pygame.draw.rect(self.screen, WHITE, self.r_state_to_rect(goal, True), LINES_WIDTH)
        for pit in self.pits:
            pygame.draw.rect(self.screen, WHITE, self.r_state_to_rect(pit), LINES_WIDTH)
            pygame.draw.rect(self.screen, RED, self.r_state_to_rect(pit, True) )
            img = pygame.image.load('images/bomb.png')
            img.convert()
            self.screen.blit(img, self.r_state_to_rect(pit))
            pygame.draw.rect(self.screen, WHITE, self.r_state_to_rect(pit, True), LINES_WIDTH)
        for wall in self.walls:
            pygame.draw.rect(self.screen, GRAY, self.r_state_to_rect(wall))
            pygame.draw.rect(self.screen, WHITE, self.r_state_to_rect(wall), LINES_WIDTH)
        pygame.display.flip()

    def r_draw_reward(self, reward, utility, done):
        top = 2*MARGIN + self.rows*CELL_SIZE
        left = MARGIN
        self.screen.blit( self.font_reward.render(f'Last reward: {reward:.2f}', True, WHITE) , (left, top) )
        self.screen.blit( self.font_reward.render(f'Utility: {utility:.2f}', True, WHITE) , (left, top+REWARD_TEXT_SIZE) )
        if done:
            self.screen.blit( self.font_reward.render(f'COMPLETED!', True, GREEN) , (left, top+2*REWARD_TEXT_SIZE) )
        pygame.display.flip()



    def r_draw_rl_metrics(self, episode, epsilon, action, explore):
        top = 2*MARGIN + self.rows*CELL_SIZE
        left = self.size_screen[0] - 2*MARGIN
        text = self.font_reward.render(f'Eposide: {episode}', True, WHITE)
        self.screen.blit( text, (left-text.get_rect().width, top) )
        text = self.font_reward.render(f'Epsilon: {epsilon:.2f}', True, WHITE)
        self.screen.blit( text, (left-text.get_rect().width, top+REWARD_TEXT_SIZE) )
        if explore:
            text = self.font_reward.render(f'Explore: {action}', True, WHITE)
        else:
            text = self.font_reward.render(f'Greedy: {action}', True, WHITE)
        self.screen.blit( text, (left-text.get_rect().width, top+2*REWARD_TEXT_SIZE) )
        pygame.display.flip()

    def r_draw_agent(self, state):
        if(state!='Terminal Diamante') and (state!='Terminal Bomba'):
            row = state[0]
            col = state[1]
            center = [MARGIN+col*CELL_SIZE+CELL_SIZE//2, MARGIN+row*CELL_SIZE+CELL_SIZE//2]
            pygame.draw.circle(self.screen, BLUE, center, AGENT_RADIUS)
            pygame.display.flip()

    def r_draw_iteration(self, iteration, flagP, flagT):
        top = 2*MARGIN + self.rows*CELL_SIZE
        left = MARGIN
        text = ""
        if(flagP=='policy'):
            if(flagT):
                text = "Policy Iteration - TESTING"
            else:
                text = "Policy Iteration"
        elif(flagP=='value'):
            if(flagT):
                text = "Value Iteration - TESTING"
            else:
                text = "Value Iteration"
        else:
            text = "Dynamic Programming"
        self.screen.blit( self.font_reward.render(text, True, WHITE) , (left, top) )
        self.screen.blit( self.font_reward.render(f'Iteration: {iteration}', True, WHITE) , (left, top+REWARD_TEXT_SIZE)  )
        pygame.display.flip()

    def r_draw_values(self, show_action=True):
        for state in self.states:
            if (state not in self.goals) and (state not in self.pits):
                rect = self.r_state_to_rect(state)
                value = self.state_values[state]
                if value>0.0:
                    color = (0.2+value*0.8)*GREEN
                else:
                    color = (0.2-value*0.8)*RED
                color = np.clip(color, 0, 255)
                pygame.draw.rect(self.screen, color, self.r_state_to_rect(state))
                pygame.draw.rect(self.screen, WHITE, self.r_state_to_rect(state), LINES_WIDTH)
                if self.policy[state]=='N':
                    corners = [ [rect[0], rect[1]],
                                [rect[0]+rect[2], rect[1]],
                                [rect[0]+rect[2]//2, rect[1]+rect[3]//6] ]
                elif self.policy[state]=='S':
                    corners = [ [rect[0], rect[1]+rect[3]],
                                [rect[0]+rect[2], rect[1]+rect[3]],
                                [rect[0]+rect[2]//2, rect[1]+5*rect[3]//6] ]
                elif self.policy[state]=='W':
                    corners = [ [rect[0], rect[1]],
                                [rect[0], rect[1]+rect[3]],
                                [rect[0]+rect[2]//6, rect[1]+rect[3]//2] ]
                else: #'E'
                    corners = [ [rect[0]+rect[2], rect[1]],
                                [rect[0]+rect[2], rect[1]+rect[3]],
                                [rect[0]+5*rect[2]//6, rect[1]+rect[3]//2] ]
                if(show_action):
                    pygame.draw.polygon(self.screen, color, corners)
                    pygame.draw.polygon(self.screen, WHITE, corners, LINES_WIDTH)
                    pygame.draw.polygon(self.screen, WHITE, corners)
                row = state[0]
                col = state[1]
                left = MARGIN + col*CELL_SIZE + CELL_SIZE//2
                top = MARGIN + row*CELL_SIZE + CELL_SIZE//2
                text = self.font_value.render(f'{value:.2f}', True, WHITE)
                left -= text.get_width()//2
                top -= text.get_height()//2
                self.screen.blit( text , (left, top) )
            else:
                if(state in self.goals):
                    value = self.state_values[state]
                else:
                    value = self.state_values[state]
                row = state[0]
                col = state[1]
                left = MARGIN + col*CELL_SIZE + CELL_SIZE//2
                top = MARGIN + row*CELL_SIZE + CELL_SIZE//2
                text = self.font_value.render(f'{value:.2f}', True, WHITE)
                left -= text.get_width()//2
                top -= text.get_height()//2
                self.screen.blit( text , (left, top) )
        pygame.display.flip()

    def r_draw_q_values(self):
        for state in self.states:
            rect = self.r_state_to_rect(state)
            row = state[0]
            col = state[1]
            for i, action in enumerate(self.state_q_values[state]):
                q_value = self.state_q_values[state][action]
                if q_value>0.0:
                    color = (0.2+q_value*0.8)*GREEN
                else:
                    color = (0.2-q_value*0.8)*RED
                if i==0: #'N'
                    corners = [ [rect[0], rect[1]],
                                [rect[0]+rect[2], rect[1]],
                                [rect[0]+rect[2]//2, rect[1]+rect[3]//2] ]
                    left = MARGIN + col*CELL_SIZE + int(0.5*CELL_SIZE)
                    top = MARGIN + row*CELL_SIZE + int(0.2*CELL_SIZE)
                elif i==1: #'S'
                    corners = [ [rect[0], rect[1]+rect[3]],
                                [rect[0]+rect[2], rect[1]+rect[3]],
                                [rect[0]+rect[2]//2, rect[1]+rect[3]//2] ]
                    left = MARGIN + col*CELL_SIZE + int(0.5*CELL_SIZE)
                    top = MARGIN + row*CELL_SIZE + int(0.8*CELL_SIZE)
                elif i==3: #'W'
                    corners = [ [rect[0], rect[1]],
                                [rect[0], rect[1]+rect[3]],
                                [rect[0]+rect[2]//2, rect[1]+rect[3]//2] ]
                    left = MARGIN + col*CELL_SIZE + int(0.2*CELL_SIZE)
                    top = MARGIN + row*CELL_SIZE + int(0.5*CELL_SIZE)

                else: #'E'
                    corners = [ [rect[0]+rect[2], rect[1]],
                                [rect[0]+rect[2], rect[1]+rect[3]],
                                [rect[0]+rect[2]//2, rect[1]+rect[3]//2] ]
                    left = MARGIN + col*CELL_SIZE + int(0.8*CELL_SIZE)
                    top = MARGIN + row*CELL_SIZE + int(0.5*CELL_SIZE)
                pygame.draw.polygon(self.screen, color, corners)
                pygame.draw.polygon(self.screen, WHITE, corners, LINES_WIDTH)
                if action==self.policy[state]:
                    text = self.font_q_value.render(f'{q_value:.2f}', True, PINK)
                else:
                    text = self.font_q_value.render(f'{q_value:.2f}', True, WHITE)
                left -= text.get_width()//2
                top -= text.get_height()//2
                self.screen.blit( text , (left, top) )

        pygame.display.flip()
