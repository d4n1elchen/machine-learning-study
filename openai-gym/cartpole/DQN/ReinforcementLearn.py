import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# Initialize random seed
np.random.seed(1)
tf.set_random_seed(1)

class DQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter_max=300,
                 replace_target_iter_min=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None):

        # Save params
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = 10000
        self.replace_target_iter_max = replace_target_iter_max
        self.replace_target_iter_min = replace_target_iter_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Initialize counters
        self.learn_step_counter = 0
        self.update_step_counter = 0
        self.memory_counter = 0

        # Initialize memory
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # Setup tf graphs and operator
        self._build_net()

        # Initialize a tf session
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Output graph
        self.summary = tf.summary.FileWriter("logs/", self.sess.graph)

    def _build_net(self):
        # Variable initializer
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)
        # w_initializer = None # Use default initializer
        # b_initializer = None

        ### Evaluation net (acturally training)
        with tf.variable_scope("eval_net"):
            # Settings
            collection_name = ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10

            # Input Layer
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_features), name="s")

            # Hidden Layer 1
            w1 = tf.get_variable("w1", shape=(self.n_features, n_l1),
                                 initializer=w_initializer, collections=collection_name)
            b1 = tf.get_variable("b1", shape=(n_l1),
                                 initializer=w_initializer, collections=collection_name)
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # Output Layer
            wo = tf.get_variable("wo", shape=(n_l1, self.n_actions),
                                 initializer=w_initializer, collections=collection_name)
            bo = tf.get_variable("bo", shape=(self.n_actions),
                                 initializer=w_initializer, collections=collection_name)
            self.q_eval = tf.matmul(l1, wo) + bo

            # Calc loss (RMS)
            self.q_target = tf.placeholder(tf.float32, shape=(None, self.n_actions), name="q_target")
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            tf.summary.scalar("loss", self.loss)

            # Build optimizer
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        ### Target net (final result, for estimate future q, update every n cycle)
        with tf.variable_scope("target_net"):
            # Settings
            collection_name = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10

            # Input Layer
            self.s_ = tf.placeholder(tf.float32, shape=(None, self.n_features), name="s_")

            # Hidden Layer 1
            w1 = tf.get_variable("w1", shape=(self.n_features, n_l1),
                                 initializer=w_initializer, collections=collection_name)
            b1 = tf.get_variable("b1", shape=(n_l1),
                                 initializer=w_initializer, collections=collection_name)
            l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # Output Layer
            wo = tf.get_variable("wo", shape=(n_l1, self.n_actions),
                                 initializer=w_initializer, collections=collection_name)
            bo = tf.get_variable("bo", shape=(self.n_actions),
                                 initializer=w_initializer, collections=collection_name)
            self.q_next = tf.matmul(l1, wo) + bo

        # Params exchange
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_ops = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # Merge summary
        self.merged_summary = tf.summary.merge_all()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))

        # Store transition
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = transition

        self.memory_counter += 1

    def get_action(self, s):
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.q_eval.eval(feed_dict={self.s: s[None, :]})
            action = np.argmax(actions_value)
        else:
            # Randomly choose action
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        # Run params exchange
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_ops)

            # increasing epsilon
            #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

            # increasing repl iter
            if self.update_step_counter % 10 == 0:
                self.replace_target_iter = max(self.replace_target_iter - 10, self.replace_target_iter_min)
                self.epsilon = self.epsilon_max / self.replace_target_iter * self.replace_target_iter_min

            print("\nLearn step:{}: Target net updated. epsilon={}, repl_iter={}\n".format(self.learn_step_counter, self.epsilon, self.replace_target_iter))

            self.update_step_counter += 1

        # Randomly select sample from transition memory
        sample_idx = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        batch_memory = self.memory[sample_idx, :]

        # Get q values for both nets
        q_eval = self.q_eval.eval(feed_dict={ self.s:  batch_memory[:,  :self.n_features] })
        q_next = self.q_next.eval(feed_dict={ self.s_: batch_memory[:, -self.n_features:] })

        # Build q_target
        q_target = q_eval.copy()
        eval_act_index = batch_memory[:, self.n_features].astype(int) # Index of actions of batch records
        reward = batch_memory[:, self.n_features + 1] # Reward of batch records
        q_target[:, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) # q_target[action] = reward + gamma*q_next

        # Train
        summary, _ = self.sess.run([self.merged_summary, self._train_op],feed_dict={
            self.q_target: q_target,
            self.s: batch_memory[:, :self.n_features]
        })

        # Save current cost
        self.summary.add_summary(summary, self.learn_step_counter)

        self.learn_step_counter += 1

        def plot_cost(self):
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(self.cost_his)), self.cost_his)
            plt.ylabel('Cost')
            plt.xlabel('training steps')
            plt.show()
