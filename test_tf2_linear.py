import unittest
from tf2_linear import Linear
import numpy as np


class TestLinearMetheods(unittest.TestCase):

    # def test_update(self):
    #
    #     model = Linear(verbose=0)
    #
    #     # test loss decreases on the same state
    #     curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
    #     reward = 0
    #     action = 0
    #     next_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
    #     done = True
    #
    #     # loss = 0.5(reward - Q(s,a))**2
    #
    #     for i in range(5):
    #         loss = model.update(curr_state, action, reward, next_state, done)
    #         if i != 0:
    #             # print(i, loss, last_loss)
    #             self.assertTrue(loss <= last_loss)
    #         last_loss = loss
    #
    #     # loss = 0.5((reward + gamma*max Q(s',a) - Q(s,a)))**2
    #
    #     curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
    #     reward = 5
    #     # action = 2 # Use Greedy action so both are max
    #     next_state = curr_state
    #     done = False
    #
    #     expected_loss = 0.5 * reward**2
    #
    #     for i in range(10):
    #         # update target network so they are the same
    #         model.update_target_weights()
    #         action, action_values = model.greedy_policy(curr_state)
    #         loss = model.update(curr_state, action, reward, next_state, done)
    #         self.assertTrue(abs(loss-expected_loss) < 0.00001)
    #
    # def test_update_target_weights(self):
    #     model = Linear(verbose=0)
    #
    #     # test loss decreases on the same state
    #     curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
    #     reward = 0
    #     next_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
    #     done = True
    #
    #     # loss = 0.5(reward - Q(s,a))**2
    #     curr_state_temp = model.preprocess(curr_state)
    #
    #     target_pred = model.target_model(curr_state_temp).numpy()
    #     reg_pred = model.model(curr_state_temp).numpy()
    #     # print(target_pred, reg_pred)
    #     self.assertTrue(np.all(target_pred == reg_pred))
    #
    #     for i in range(50):
    #         action = np.random.randint(5)
    #         loss = model.update(curr_state, action, reward, next_state, done)
    #
    #     target_pred = model.target_model(curr_state_temp).numpy()
    #     reg_pred = model.model(curr_state_temp).numpy()
    #
    #     self.assertFalse(np.any(target_pred == reg_pred))
    #     model.update_target_weights()
    #
    #     target_pred = model.target_model(curr_state_temp).numpy()
    #     self.assertTrue(np.all(target_pred == reg_pred))
    #

    def test_greedy_policy(self):
        curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        model = Linear(espilon=0, verbose=0)

        for i in range(10):
            action, action_values = model.greedy_policy(curr_state)
            self.assertEqual(action, np.argmax(action_values))

    def test_preprocess(self):

        model = Linear(espilon=0, verbose=0)

        # Test Zero centered and normalized
        for i in range(10):

            curr_state = np.random.randint(
                0, 255, [25, 5, 5, 1], dtype=np.uint16)
            curr_state = model.preprocess(curr_state)

            channel_mean = np.average(curr_state, axis=(1, 2))
            self.assertTrue(abs(np.max(channel_mean)) < 0.00001)
            mean = np.average(curr_state)
            max = np.amax(curr_state)
            self.assertTrue(abs(mean) < 0.00001)
            self.assertEqual(max, 1.0)

        # Test correctly meaning
        noise_mat = np.random.uniform(size=[1, 5, 5, 1])
        curr_state = np.ones(shape=[1, 5, 5, 1]) + noise_mat
        for i in range(10):
            curr_state = np.concatenate(
                [curr_state, np.ones(shape=[1, 5, 5, 1])*(i+1) + noise_mat])
        self.assertEqual(curr_state.shape[0], 11)

        curr_state = model.preprocess(curr_state)

        for i in range(11):
            img = curr_state[i, :, :, :]
            img_total = np.sum(img)
            self.assertTrue(abs(img_total) < 1e-5)

            img_max = np.amax(img)
            self.assertEqual(img_max, 1)

    def test_replay_update_target_weights(self):
        buffer_len = 50
        model = Linear(replay_buffer_len=buffer_len, verbose=0)

        # test loss decreases on the same state
        curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        reward = 0
        next_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        done = True

        for i in range(buffer_len):  # must use a different action to update all parts of network
            action = np.random.randint(5)
            model.replay_fill(curr_state, action, reward, next_state, done)

        curr_state_temp = model.preprocess(curr_state)
        target_pred = model.target_model(curr_state_temp).numpy()
        reg_pred = model.model(curr_state_temp).numpy()
        # print(target_pred, reg_pred)
        self.assertTrue(np.all(target_pred == reg_pred))

        for i in range(20):
            loss = model.replay_update()

        target_pred = model.target_model(curr_state_temp).numpy()
        reg_pred = model.model(curr_state_temp).numpy()

        self.assertFalse(np.any(target_pred == reg_pred))
        model.update_target_weights()

        target_pred = model.target_model(curr_state_temp).numpy()
        self.assertTrue(np.all(target_pred == reg_pred))

    def test_policy(self):

        # in Greedy Case
        curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        model = Linear(espilon=0, verbose=0)

        for i in range(10):
            action, action_values = model.policy(curr_state)
            self.assertEqual(action, np.argmax(action_values))

        # In Pure Exploratory Case
        curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        model = Linear(espilon=1, verbose=0)

        iter = 50
        num_actions = 5
        max_count = (iter / num_actions) * 2
        count = 0
        for i in range(iter):
            action, action_values = model.policy(curr_state)

            if action == np.argmax(action_values):
                count += 1

        self.assertLess(count, max_count, msg="Picking Greedy action to often, \
        mabye run one more time to account for random sampling")

    def test_replay_fill(self):

        buffer_len = 10
        model = Linear(replay_buffer_len=buffer_len, verbose=0)

        # test loss decreases on the same state
        curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        action = 1
        reward = 0
        next_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        done = True

        # Check increments
        model.replay_fill(curr_state, action, reward, next_state, done)
        self.assertTrue(model.replay_buffer.num_in_buffer == 1)
        model.replay_fill(curr_state, action, reward, next_state, done)
        model.replay_fill(curr_state, action, reward, next_state, done)
        model.replay_fill(curr_state, action, reward, next_state, done)
        self.assertTrue(model.replay_buffer.num_in_buffer == 4)

        # Check Overflow
        for i in range(50):
            model.replay_fill(curr_state, action, reward, next_state, done)
        self.assertTrue(model.replay_buffer.num_in_buffer == buffer_len)

    def test_replay_sample(self):
        buffer_len = 10
        model = Linear(replay_buffer_len=buffer_len, verbose=0)

        # Test reading from buffer with done = False
        curr_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        action = 1
        reward = 0
        next_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        done = False
        next_next_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)

        model.replay_fill(curr_state, action, reward, next_state, done)
        model.replay_fill(next_state, action, reward, next_next_state, done)

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = model.replay_buffer.sample(
            1)

        self.assertTrue(np.all(curr_state == obs_batch[0, :, :, :]))
        self.assertTrue(action == act_batch[0])
        self.assertTrue(reward == rew_batch[0])
        self.assertTrue(done == done_mask[0])

        self.assertTrue(np.all(next_state ==
                               next_obs_batch[0, :, :, :]))

        # Agian, with done True
        buffer_len = 10
        model = Linear(replay_buffer_len=buffer_len, verbose=0)

        curr_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        action = 1
        reward = 0
        next_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        done = True
        next_next_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        model.replay_fill(curr_state, action, reward, curr_state, done)
        model.replay_fill(curr_state, action, reward, next_state, done)

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = model.replay_buffer.sample(
            1)

        self.assertTrue(np.all(curr_state == obs_batch[0, :, :, :]))
        self.assertTrue(action == act_batch[0])
        self.assertTrue(reward == rew_batch[0])
        self.assertTrue(done == done_mask[0])
        self.assertTrue(np.all(curr_state == next_obs_batch[0, :, :, :]))

        # Check it updates fully
        curr_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        action = 0
        reward = 5
        next_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        done = False

        for i in range(buffer_len):
            model.replay_fill(curr_state, action, reward, next_state, done)

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = model.replay_buffer.sample(
            9)

        for i in range(buffer_len-1):
            self.assertTrue(np.all(curr_state == obs_batch[i, :, :, :]))
            self.assertTrue(action == act_batch[i])
            self.assertTrue(reward == rew_batch[i])
            self.assertTrue(done == done_mask[i])
            self.assertTrue(np.all(curr_state == next_obs_batch[i, :, :, :]))

    def test_replay_update(self):
        buffer_len = 50
        model = Linear(replay_buffer_len=buffer_len,
                       learning_rate=0.001, verbose=0)

        # test loss decreases on the same state
        curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        action = 1
        reward = 0
        next_state = np.random.randint(0, 50, [5, 5, 1], dtype=np.uint16)
        done = True

        # prepopulate buffer
        for i in range(buffer_len):
            model.replay_fill(curr_state, action, reward, next_state, done)

        # loss = 0.5(reward - Q(s,a))**2

        for i in range(5):
            loss = model.replay_update()
            if i != 0:
                self.assertTrue(loss <= last_loss)
            last_loss = loss

        # loss = 0.5((reward + gamma*max Q(s',a) - Q(s,a)))**2

        curr_state = np.random.randint(0, 255, [5, 5, 1], dtype=np.uint16)
        reward = 5
        # action = 2 # Use Greedy action so both are max
        next_state = curr_state
        done = False

        # prepopulate buffer
        for i in range(buffer_len):
            model.replay_fill(curr_state, action, reward, next_state, done)

        expected_loss = 0.5 * reward**2

        for i in range(10):
            # update target network so they are the same
            model.update_target_weights()
            action, action_values = model.greedy_policy(curr_state)
            loss = model.update(curr_state, action, reward, next_state, done)
            self.assertTrue(abs(loss-expected_loss) < 0.00001)


if __name__ == '__main__':
    unittest.main()
