import torch
import numpy as np
import envs.base_env as base_env

class Viper(base_env.BaseEnv):
    def __init__(self, _sks):
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

        self.sks = _sks
        self.gcd = self.compute_gcd(2.5, self.sks, 0)
        self.gcd_roll = 10.0
        self.action_lock_duration = 0.7 # It's closer to 0.6 in reality, which does allow triple weaving but near impossible in most cases.
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
        self.gcd = self.compute_gcd(2.5, self.sks, 0)
        self.gcd_roll = 10.0
        self.action_lock_duration = 0.7 # It's closer to 0.6 in reality, which does allow triple weaving but near impossible in most cases.
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
        return state.shape[0]

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
                # step time forward to the next gcd spot, tick buffs as needed
                time_malus = self.valid_action(2.5)
                bonus = 0
                if self.honed_steel > 0:
                    self.honed_steel = 0
                    bonus = 100
                # we apply buffs here to prevent premature ticking on them
                self.honed_reavers = 60.0
                self.filler_stage = 1

                # now time step to the next free animation slot, tick buffs as needed
                self.action_lock(self.action_lock_duration)
                action_success = True
                action_reward = 200 + bonus
        elif action_name == 'reaving_fangs':
            if self.filler_stage == 0:
                time_malus = self.valid_action()
                bonus = 0
                if self.honed_reavers > 0:
                    self.honed_reavers = 0
                    bonus = 100
                self.honed_steel = 60
                self.filler_stage = 1

                self.action_lock(self.action_lock_duration)
                action_success = True
                action_reward = 200 + bonus
        elif action_name == 'hunters_sting':
            if self.filler_stage == 1:
                time_malus = self.valid_action()
                self.hunters_instinct = 40.0
                self.filler_stage = 2

                self.action_lock(self.action_lock_duration)
                action_success = True
                action_reward = 300

        # on fail / bad action, step forward 100 ms
        if self.hunters_instinct > 0:
            action_reward = action_reward * 1.1
        if not action_success:
            time_malus = self.invalid_action()

        damage = self.compute_damage(action_reward)
        reward = action_reward - time_malus
        return reward, damage

    def invalid_action(self):
        # Hard lock for 100 ms to punish incorrect flow.
        delta_time = 0.100
        return self.time_step(delta_time)
    
    def compute_gcd(self, base_gcd, sks, haste):
        # from https://github.com/xiv-gear-planner/gear-planner/blob/master/packages/xivmath/src/xivmath.ts
        sks = max(420, sks) # clamp to 420 min
        x = np.floor(130 * (sks - 420) / 2780)
        x = np.floor((1000 - x) * base_gcd) * (100 - haste) / 1000
        x = np.floor(x) / 100
        return x

    # handle gcd timing for valid gcd actions
    def valid_action(self, gcd_lock=2.5):
        delta_time = 0
        total_time = 0
        # Roll the gcd forward to the next possible time
        delta_time = max(self.gcd - self.gcd_roll, 0) # accounts for clipping
        total_time += self.time_step(delta_time)
        self.gcd_roll = 0.0

        # if we have haste buff then adjust gcd lock
        self.gcd = self.compute_gcd(gcd_lock, self.sks, 15 if self.swiftscaled > 0 else 0)
        return total_time  
    
    # handle all animation action locks
    def action_lock(self, lock_time):
        # roll gcd for action lock time, clip gcd if necessary
        delta_time = lock_time
        return self.time_step(delta_time)
    
    def compute_damage(self, potency):
        variance = np.random.uniform(0.95, 1.05)
        damage = np.floor(potency * variance * 100) / 100
        return damage

    # subtract 10 potency per 100 ms of action
    def time_step(self, delta_time):
        self.gcd_roll += delta_time
        self.time += delta_time
        # TODO handle all buff timers and update here
        self.honed_reavers = max(0, self.honed_reavers - delta_time)
        self.honed_steel = max(0, self.honed_steel - delta_time)
        self.hunters_instinct = max(0, self.hunters_instinct - delta_time)
        self.swiftscaled = max(0, self.swiftscaled - delta_time)

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
            self.hunters_instinct,
            self.swiftscaled
        ]
        return torch.tensor(_state, dtype=torch.float32, device=torch.device('cpu'))
    
    def is_done(self):
        return False