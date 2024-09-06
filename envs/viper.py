import torch

class Viper:
    def __init__(self, _base_gcd=2.5):
        print('b123')

        # action list
        self.actions = [
            ('nothing', ''),
            ('steel_fangs', 'gcd'),
            ('reaving_fangs', 'gcd'),
            ('hunters_sting', 'gcd'),
            ('swiftskins_sting', 'gcd'),
            ('flanksting_strike', 'gcd'),
            ('flanksbane_fang', 'gcd'),
            ('hindsting_strike', 'gcd'),
            ('hindsbane_fang', 'gcd'),
            ('writhing_snap', 'gcd'),
            ('death_rattle', 'ogcd'),
        ]

        self.base_gcd = _base_gcd
        self.gcd = self.base_gcd
        self.gcd_roll = 10.0
        self.action_lock_duration = 0.7 # It's closer to 0.6 in reality, which does allow triple weaving but near impossible in most cases.
        self.action_lock = 0.0
        self.time = 0.0

        # buffs
        self.filler_stage = 0
        self.honed_reavers = 0.0
        self.honed_steel = 0.0
        self.hunters_instinct = 0.0
        self.swiftscaled = 0.0
        # rotation stuff buffs
        self.hindstung_venom = 0.0
        self.hindsbane_venom = 0.0
        self.flanksbane_venom = 0.0
        self.flankstung_venom = 0.0

    # Need something more elegant
    def reset_env(self):
        self.gcd = self.base_gcd
        self.gcd_roll = 10.0
        self.action_lock_duration = 0.7 # It's closer to 0.6 in reality, which does allow triple weaving but near impossible in most cases.
        self.action_lock = 0.0
        self.time = 0.0

        # buffs
        self.filler_stage = 0
        self.honed_reavers = 0.0
        self.honed_steel = 0.0
        self.hunters_instinct = 0.0
        self.swiftscaled = 0.0
        # rotation stuff buffs
        self.hindstung_venom = 0.0
        self.hindsbane_venom = 0.0
        self.flanksbane_venom = 0.0
        self.flankstung_venom = 0.0


    def get_max_actions(self):
        return len(self.actions)
    
    def get_state_shape(self):
        state = self.state()
        return state.shape

    def step(self, action: int):
        if action < 0 or action > self.get_max_actions()-1:
            print(f'Invalid action {action} chosen')
            return
        action_name = self.actions[action][0]
        print(f'Taking action {action_name}')
        action_reward = 0.0
        time_malus = 0.0
        action_time = 0.0
        action_success = False
        # All buffs given from actions should be applied here, consume or reset depending on interactions
        #  with the rest of the toolkit
        if action_name == 'steel_fangs':
            if self.filler_stage == 0:
                bonus = 0
                if self.honed_steel > 0:
                    self.honed_steel = 0
                    bonus = 100
                self.honed_reavers = 60.0
                self.filler_stage = 1

                time_malus = self.valid_action(self.action_lock_duration)
                action_success = True
                action_reward = 200 + bonus
        elif action_name == 'reaving_fangs':
            if self.filler_stage == 0:
                bonus = 0
                if self.honed_reavers > 0:
                    self.honed_reavers = 0
                    bonus = 100
                self.honed_steel = 60
                self.filler_stage = 1

                time_malus = self.valid_action(self.action_lock_duration)
                action_success = True
                action_reward = 200 + bonus
        elif action_name == 'hunters_sting':
            if self.filler_stage == 1:
                self.hunters_instinct = 40.0
                self.filler_stage = 2

                time_malus = self.valid_action(self.action_lock_duration)
                action_success = True
                action_reward = 300

        # on fail / bad action, step forward 100 ms
        if self.hunters_instinct > 0:
            action_reward = action_reward * 1.1
        if not action_success:
            time_malus = self.invalid_action()

        reward = action_reward - time_malus
        return reward

    def invalid_action(self):
        # Hard lock for 100 ms to punish incorrect flow.
        delta_time = 0.100
        return self.time_step(delta_time)

    def valid_action(self, lock_time, is_ogcd = False):
        delta_time = 0
        total_time = 0
        if not is_ogcd:
            # Roll the gcd forward to the next possible time
            delta_time = max(self.gcd - self.gcd_roll, 0) # accounts for clipping
            total_time += self.time_step(delta_time)
            self.gcd_roll = 0.0
        # roll gcd for action lock time, clip gcd if necessary
        delta_time = lock_time
        return total_time + self.time_step(delta_time)
    
    def compute_damage(self, potency):
        pass

    # subtract 10 potency per 100 ms of action
    def time_step(self, delta_time):
        self.gcd_roll += delta_time
        self.time += delta_time
        # TODO handle all buff timers and update here
        self.honed_reavers = max(0, self.honed_reavers - delta_time)
        self.honed_steel = max(0, self.honed_steel - delta_time)

        return delta_time / 0.100 # every 100 ms incurs 1 potency cost to punish hard clipping.

    def state(self):
        # We should return the state as a tensor for ease of use
        _state = [
            self.time,
            self.gcd,
            self.gcd_roll,
            self.filler_stage,
            self.honed_reavers,
            self.honed_steel,
            self.hunters_instinct
        ]
        return torch.tensor(_state, dtype=torch.float32, device=torch.device('cpu'))