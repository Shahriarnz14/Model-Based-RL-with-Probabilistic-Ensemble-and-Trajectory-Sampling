import os
import tensorflow as tf
import numpy as np
import gym
import copy


class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """

        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
        :param use_mpc: Whether to use only the first action of a planned trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """
        self.env = env
        self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
        self.num_particles = num_particles
        self.plan_horizon = plan_horizon
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        
        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters

        self.mu = np.zeros((self.action_dim * self.plan_horizon,))
        self.sigma = 0.5 ** 2 * np.ones((self.plan_horizon * self.action_dim,))

        self.action_init = np.zeros((self.action_dim,))
        self.sigma_init = 0.5 ** 2 * np.ones((self.plan_horizon * self.action_dim,))

        self.mean_init = np.zeros((self.action_dim * self.plan_horizon,))
        self.cov_init = 0.5 * np.eye(self.plan_horizon * self.action_dim)

        self.train_inputs = np.array([]).reshape(0, self.action_dim + self.state_dim)
        self.train_targets = np.array([]).reshape([0, self.state_dim])
        self.is_trained = False


    def obs_cost_fn(self, state): # add goals
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord


    def trajectory_sampling_1(self, N, P, T):
        self.S = np.zeros((P, T))
        for p in range(P):
            self.S[p, :] = np.random.randint(N, size=T)

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        states = np.reshape(states, [self.popsize, self.num_particles, self.state_dim])
        network_inputs = [np.zeros((self.popsize * self.num_particles // self.num_nets, self.state_dim + self.action_dim)) for _ in range(self.num_nets)]
        counter_of_network = np.zeros(self.num_nets, dtype=int)

        assignment = {}
        for pop_idx in range(self.popsize):
            # network_assignment = np.random.randint(high=self.num_nets, size=self.num_particles)
            network_assignment = np.random.permutation(self.num_particles) * self.num_nets // self.num_particles  # More Balanced
            for particle_idx in range(self.num_particles):
                state = states[pop_idx, particle_idx]
                network_idx = network_assignment[particle_idx]
                network_inputs[network_idx][counter_of_network[network_idx], :] = np.concatenate((state, actions[pop_idx, :]), axis=-1)
                assignment[(network_idx, counter_of_network[network_idx])] = (pop_idx, particle_idx)
                counter_of_network[network_idx] += 1
            # end_for
        # end_for

        outputs = [self.model.models[n].predict(network_inputs[n]) for n in range(self.num_nets)]
        dynamics_difference = np.zeros((self.popsize, self.num_particles, self.state_dim))

        for n in range(self.num_nets):
            mean = outputs[n][:, :self.state_dim]
            std = np.exp(outputs[n][:, self.state_dim:] * 0.5)
            # predictions = mean + np.random.normal(size=mean.shape) * np.sqrt(var)
            predictions = mean + np.random.normal(size=mean.shape) * std
            for assign_idx in range(predictions.shape[0]):
                population_idx, particle_idx = assignment[(n, assign_idx)]
                dynamics_difference[population_idx, particle_idx, :] = predictions[assign_idx, :]
            # end_for
        # end_for

        new_states = states + dynamics_difference
        # new_states = dynamics
        new_states = np.reshape(new_states, (-1, self.state_dim))

        return new_states

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        new_states = [self.env.get_nxt_state(state, action) for state, action in zip(states, actions)]
        return new_states

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """


        new_train_inputs = []
        new_train_targets = []

        # print(np.shape(obs_trajs))
        # print(np.shape(acs_trajs))
        # print('======')
        for obs, acs in zip(obs_trajs, acs_trajs):
            # print(np.shape(obs))
            # print(np.shape(acs))
            new_train_inputs.append(np.concatenate([obs[:-1, :-2], acs], axis=-1))
            new_train_targets.append(np.concatenate([obs[1:, :-2] - obs[:-1, :-2]], axis=-1))
            # new_train_inputs.append(np.concatenate([obs[:-1], acs], axis=-1))


        self.train_inputs = np.concatenate([self.train_inputs] + new_train_inputs, axis=0)
        self.train_targets = np.concatenate([self.train_targets] + new_train_targets, axis=0

        # Train the model
        # print(new_train_inputs)
        # print(np.shape(new_train_targets))
        self.model.train(self.train_inputs, self.train_targets, epochs=epochs)
        self.is_trained = True

    def reset(self):
        self.mu = np.zeros((self.action_dim*self.plan_horizon,))
        self.sigma = 0.5*np.eye(self.action_dim*self.plan_horizon)

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        self.goal = state[-2:]
        self.curr_state = state[0:-2]

        if not self.is_trained and not self.use_gt_dynamics:
            return np.random.uniform(-1, 1, self.action_dim) 

        internal_t = t % self.plan_horizon
        if self.use_mpc and self.use_random_optimizer:
            self.mu = self.random_optimizer().flatten()
            action = self.mu[0:2]
            self.mu = np.append(self.mu[self.action_dim:], self.action_init)
            return action
        elif self.use_mpc:
            self.mu = self.cem_optimizer(self.mu, self.sigma_init)
            action = self.mu[0:2]
            self.mu = np.append(self.mu[self.action_dim:], self.action_init)
            return action
        elif self.use_random_optimizer:
            if internal_t == 0:
                self.mu = self.random_optimizer().flatten()
            return self.mu[self.action_dim * internal_t:self.action_dim * internal_t + 2]
        else:
            if internal_t == 0:
                self.mu = self.cem_optimizer(self.mu, self.sigma_init)
            return self.mu[self.action_dim * internal_t:self.action_dim * internal_t + 2]
        # end_if

    def cem_optimizer(self, mu, sigma):

        for iter_idx in range(self.max_iters):
            action_sequences = np.random.multivariate_normal(mean=mu,
                                                             cov=sigma*np.eye(self.action_dim*self.plan_horizon),
                                                             size=self.popsize)
            action_sequences = np.clip(action_sequences, -1, 1)

            total_costs = np.zeros([self.popsize, self.num_particles])
            curr_state = np.tile(self.curr_state[None], [self.popsize * self.num_particles, 1])

            ac_squences = np.reshape(action_sequences, [-1, self.plan_horizon, self.action_dim])
            ac_squences = np.transpose(ac_squences, [1, 0, 2])

            for time_step in range(self.plan_horizon):
                actions_current = ac_squences[time_step]
                next_state = self.predict_next_state(curr_state, actions_current)
                costs_diff = np.apply_along_axis(self.obs_cost_fn, -1, next_state)
                costs_diff = costs_diff.reshape((self.popsize, -1))

                total_costs += costs_diff
                curr_state = next_state
            # end_for

            costs_population = np.mean(np.where(np.isnan(total_costs), np.ones_like(total_costs) * 1e6, total_costs), axis=1)
            elites = action_sequences[np.argpartition(costs_population, self.num_elites)[:self.num_elites]]

            mu = np.mean(elites, axis=0)
            sigma = np.var(elites, axis=0)
        # end_for

        return mu

    def random_optimizer(self):

        prev_cost = np.inf

        init_mean = self.mean_init # np.zeros(self.action_dim * self.plan_horizon)
        init_cov = self.cov_init  # 0.5 * np.eye(self.action_dim * self.plan_horizon)

        # print(init_cov)

        for iter_idx in range(self.max_iters):
            action_sequences = np.random.multivariate_normal(mean=init_mean,
                                                             cov=init_cov,
                                                             size=self.popsize)
            # costs_pop = np.zeros((self.popsize,))
            action_sequences = np.clip(action_sequences, -1, 1)

            total_costs = np.zeros([self.popsize, self.num_particles])
            curr_state = np.tile(self.curr_state[None], [self.popsize * self.num_particles, 1])

            ac_seqs = np.reshape(action_sequences, [-1, self.plan_horizon, self.action_dim])
            ac_seqs = np.transpose(ac_seqs, [1, 0, 2])

            for time_step in range(self.plan_horizon):
                actions_current = ac_seqs[time_step]
                next_state = self.predict_next_state(curr_state, actions_current)
                costs_diff = np.apply_along_axis(self.obs_cost_fn, -1, next_state)
                costs_diff = costs_diff.reshape((self.popsize, -1))

                total_costs += costs_diff
                curr_state = next_state
            # end_for

            costs_pop = np.mean(np.where(np.isnan(total_costs), 1e6 * np.ones_like(total_costs), total_costs), axis=1)

            # elites = action_sequences[np.argpartition(costs_pop, self.num_elites)[:self.num_elites]]
            # elite_index = np.argsort(costs_pop)[:1]
            # elite_cost = costs_pop[elite_index]
            # elite = action_sequences[np.argsort(costs_pop)[:1]]

            elite_index = np.argmin(costs_pop)
            elite_cost = costs_pop[elite_index]
            elite = action_sequences[elite_index]

            if elite_cost < prev_cost:
                prev_cost = elite_cost
                mu = elite
        # end_for

        return mu
