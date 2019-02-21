import gym
import random
import numpy as np
import tensorflow as tf
import csv
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

#--------------------------------------------------------------------------------------------------------

EPISODES = 50000

#--------------------------------------------------------------------------------------------------------

class TestAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.no_op_steps = 20

        self.model = self.build_model()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()

        return model

    def get_action(self, history):
        if np.random.random() < 0.01:
            return random.randrange(3)
        history = np.float32(history / 255.0)
        q_value = self.model.predict(history)
        return np.argmax(q_value[0])

    def load_model(self, filename):
        self.model.load_weights(filename)

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

#--------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    env = gym.make('MsPacmanDeterministic-v4')
    agent = TestAgent(action_size=9)
    agent.load_model("./saved_model/pacman_dqn/pacman_dqn.h5")

    with open (b"./data_csv/pacman_dqn_play_game_data.csv","w") as csv_file:
        writer = csv.writer(csv_file,delimiter=',')
        writer.writerow(["Episode:","Score:"])


        for e in range(EPISODES):
            done = False
            dead = False

            step, score, start_life = 0, 0, 5
            observe = env.reset()



            for _ in range(random.randint(1, agent.no_op_steps)):
                observe, _, _, _ = env.step(1)

            state = pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                env.render()
                step += 1

                action = agent.get_action(history)

                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                if dead:
                    real_action = 1
                    dead = False

                observe, reward, done, info = env.step(real_action)

                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward

                history = next_history

                if done:
                    print("Episode:", e, "  Score:", score)
                    writer.writerow([e,score])
