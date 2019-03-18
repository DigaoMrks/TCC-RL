import gym
import random
import numpy as np
import tensorflow as tf
import csv
import datetime
import os

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

#--------------------------------------------------------------------------------------------------------

# Nome

S_NAME = 'LR1-8'

GAME = 'Pacman_'
MODEL = '_DQN'
NAME = GAME+S_NAME+MODEL

#--------------------------------------------------------------------------------------------------------

# GAME SETTINGS
ENV_NAME = 'MsPacmanDeterministic-v4' # Nome do jogo
ACTION = 9 # Quantidade de possíveis ações no jogo. 'do nothing', também é uma ação
                # Deterministic-v4 version use 4 actions (Procurei no artigo e realmente usa 4 ações mas não sei o porque, <, >, 'do nothing')
                #Segundo a biblioteca GYM: https://github.com/openai/gym/wiki/Table-of-environments
RENDER = False # Renderizar ou não o treinamento
LOAD_SAVED_MODEL = False # Load no modelo já treinado para continuar o treinamento

#------------------

# Environment Settings
FRAME_WIDTH = 84 # Número de pixels da largura
FRAME_HEIGHT = 84 # Número de pixels da altura
STATE_LENGTH = 4 # Número de frames 'juntos', nesse caso 4 frames reais é 1 frame para a rede

#------------------

# Epsilon Parameters
INITIAL_EPSILON = 1.0 # Valor inicial de Epsilon (Exploration)(Sabe-se pouco do ambiente)
FINAL_EPSILON = 0.1 #Valor final de Epsilon (Exploitation)(Sabe-se muito do ambiente)
EXPLORATION_STEPS = 1000000 # Número de passos que o valor inicial de epsilon é linearmente alinhado com o valor final de epsilon

#------------------

# Training Parameters
EPISODES = 1001 #Número de episódios/epocas(epoch)
BATCH_SIZE = 32 # Minimo Batch size
TARGET_UPDATE_INTERVAL = 10000  # Frequência na qual a rede é atualizada
GAMMA = 0.99 # Valor do Discount factor
NUM_REPLAY_MEMORY = 400000 # Número máximo de replay memory que o agente usa para trainamento
NO_OP_STEPS = 30 # Número de ações de 'do nothing' possíveis para o agente no início do episódio

LEARNING_RATE = 0.00000001 # Learing rate usado pelo RMSProp (Não sei explicar)
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

#--------------------------------------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, action_size):
        self.render = RENDER
        self.load_model = LOAD_SAVED_MODEL

        # Environment Settings
        self.state_size = (FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH)
        self.action_size = action_size # Número de ações possíveis no jogo, + 'do nothing' (Breakout = <, >, 'do nothing' = 3)

        # Epsilon Parameters
        self.epsilon = 1.
        self.epsilon_decay_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        # Training Parameters
        self.train_start = EPISODES
        self.update_target_rate = TARGET_UPDATE_INTERVAL
        self.memory = deque(maxlen=NUM_REPLAY_MEMORY)
        self.no_op_steps = NO_OP_STEPS

        # Build Model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('trained/'+NAME+'/summary/', self.sess.graph)
        #self.summary_writer = tf.summary.FileWriter('summary/pacman_dqn/pacman_dqn_padrao_2k', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model: self.model.load_weights("./trained/"+NAME+"/saved_model/"+NAME+".h5")
        #if self.load_model: self.model.load_weights("./saved_model/pacman_dqn/pacman_dqn_padrao_2k.h5")

#--------------------------------------------------------------------------------------------------------
    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error

    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = Adam(lr=LEARNING_RATE, epsilon=MIN_GRAD)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

#--------------------------------------------------------------------------------------------------------

    # Cria uma rede neural igual ao do artigo (Human-level control through deep reinforcement learning)
    # A rede criada tem como entrada a 'imagem' 84x84x4 (84 largura x 84 altura x 4 frames)
    # De acordo com o artigo:
    # The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity.
    # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
    # This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
    # The final hidden layer is fully-connected and consists of 512 rectifier units.
    # The output layer is a fully-connected linear layer with a single output for each valid action.

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

#--------------------------------------------------------------------------------------------------------

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > FINAL_EPSILON: self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, BATCH_SIZE)

        history = np.zeros((BATCH_SIZE, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((BATCH_SIZE, self.state_size[0], self.state_size[1], self.state_size[2]))
        target = np.zeros((BATCH_SIZE,))
        action, reward, dead = [], [], []

        for i in range(BATCH_SIZE):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(BATCH_SIZE):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + GAMMA * np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, name):
        self.model.save_weights(name)

#--------------------------------------------------------------------------------------------------------

    #Sumário de operadores do Tensorboard

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

#--------------------------------------------------------------------------------------------------------

    # Transforma a imagem de entrada na rede neural de 210*160*3(colorido) -> 84*84(mono)
    # Isso reduz o tamanho do replay de memória para agilizar o processo

def pre_processing(observe):
    processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

#--------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = DQNAgent(action_size=ACTION)
    frames=0

    time = datetime.datetime.now()

    if not os.path.exists('./trained/'+NAME+'/data_csv/'):
        os.makedirs('./trained/'+NAME+'/data_csv/')

    if not os.path.exists('./trained/'+NAME+'/saved_model/'):
        os.makedirs('./trained/'+NAME+'/saved_model/')

    # Salva em um csv todos os dados do treinamento (segurança pois estava com problema para usar o tensorboard)
    with open ("./trained/"+NAME+"/data_csv/"+NAME+".csv","w") as csv_file:
    #with open (b"./data_csv/pacman_dqn/pacman_dqn_padrao_2k.csv","w") as csv_file:
        writer = csv.writer(csv_file,delimiter=',')
        writer.writerow(['EPISODES','BATCH_SIZE','GAMMA','NO_OP_STEPS','LEARNING_RATE','MIN_GRAD'])
        writer.writerow([EPISODES,BATCH_SIZE,GAMMA,NO_OP_STEPS,LEARNING_RATE,MIN_GRAD])
        writer.writerow(['Date/Time Start',time])
        writer.writerow(['Episode','Score','Mem Lenght','Epsilon','Global Step','Average_q','Average_Loss','Frames'])

        scores, episodes, global_step = [], [], 0

        for e in range(EPISODES):
            done = False
            dead = False
            # 1 episode = 5 lives
            step, score, start_life = 0, 0, 5
            observe = env.reset()

            #Para não ocorrer sub-optimal, a DeepMind propõe:
            #Durante o inicio de um episódio, faça uma ação 'do nothing'
            for _ in range(random.randint(1, agent.no_op_steps)): observe, _, _, _ = env.step(1)

            # No inicio do episódio, não existe frame anterior
            # Então, é copiado o estado inicial para fazer o histórico
            state = pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                if agent.render:
                    env.render()
                global_step += 1
                step += 1
                frames+=1

                # get action for the current history and go one step in environment
                action = agent.get_action(history)
                # change action to real_action
                if action == 0: real_action = 1
                elif action == 1: real_action = 2
                else: real_action = 3

                observe, reward, done, info = env.step(real_action)
                # pre-process the observation --> history
                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

                # Se o agente erre a bola, o agente "morre" --> MAS o episodio não terminou
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                reward = np.clip(reward, -1., 1.)

                # Salva os valores de <s, a, r, s'> para o replay memory
                agent.replay_memory(history, action, reward, next_history, dead)
                # every some time interval, train model
                agent.train_replay()
                # update the target model with model
                if global_step % agent.update_target_rate == 0: agent.update_target_model()

                score += reward

                # Se o agente morrer, da um reset na história
                if dead:
                    dead = False
                else:
                    history = next_history

                # Caso o Episódio acabou, plota os dados do episódio na tela e no csv
                if done:
                    if global_step > agent.train_start:
                        stats = [score, agent.avg_q_max / float(step), step, agent.avg_loss /     float(step)]
                        for i in range(len(stats)):
                            agent.sess.run(agent.update_ops[i], feed_dict={
                                agent.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, e + 1)

                    # Mostra na tela todos os dados
                    print("episode:", e, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon,
                          "  global_step:", global_step, "  average_q:",
                          agent.avg_q_max / float(step), "  average loss:",
                          agent.avg_loss / float(step), "e%", (e % 1000), "Frames:", frames)

                    # Salva em um csv todos os dados do treinamento (segurança pois estava com problema para usar o tensorboard)
                    writer.writerow([e,score,len(agent.memory),agent.epsilon,global_step,agent.avg_q_max / float(step),agent.avg_loss / float(step), frames])

                    agent.avg_q_max, agent.avg_loss = 0, 0

            # Salva o modelo de 1000 em 1000 iterações (AVALIAR SE É MELHOR SALVAR POR EPOCA OU POR FRAME)
            if e % 1000 == 0:
                agent.model.save_weights("./trained/"+NAME+"/saved_model/"+NAME+".h5")
                print("MODEL SAVED in: "+"./trained/"+NAME+"/saved_model/"+NAME+".h5")
                #agent.model.save_weights("./saved_model/pacman_dqn/pacman_dqn_padrao_2k.h5")
                #print("MODEL SAVED in: saved_model/pacman_dqn/pacman_dqn_padrao_2k.h5")

        time_end = datetime.datetime.now()
        writer.writerow(['Date/Time End',time_end])
    csv_file.close()
