import torch
import numpy as np
import envs.base_env as base_env

class Viper(base_env.BaseEnv):
    def __init__(self, _sks):
        super(Viper, self).__init__()
        print('b123')

        # action list
        self.actions = [
            #('nothing', ''),               # 
            ('steel_fangs', 'gcd'),         # 0
            ('reaving_fangs', 'gcd'),       # 1
            ('hunters_sting', 'gcd'),       # 2
            ('swiftskins_sting', 'gcd'),    # 3
            ('flanksting_strike', 'gcd'),   # 4
            ('flanksbane_fang', 'gcd'),     # 5
            ('hindsting_strike', 'gcd'),    # 6
            ('hindsbane_fang', 'gcd'),      # 7
            ('writhing_snap', 'gcd'),       # 8 - INOP
            ('death_rattle', 'ogcd'),       # 9
        ]

        self.sks = _sks
        self.reset_env()

    # Need something more elegant
    def reset_env(self, _sks=None):
        if _sks is not None:
            self.sks = _sks
        self.gcd = self.compute_gcd(2.5, self.sks, 0)
        self.gcd_roll = 10.0 # Initially set to ten, guarantees the first action will go instantly
        self.action_lock_duration = 0.7 # It's closer to 0.6 in reality, which does allow triple weaving but near impossible in most cases.
        self.time = 0.0

        # buffs
        # 0: Initial
        # 1: Second Stage
        # 2: Flanks <- From Hunters
        # 3: Rears  <- From Swift
        self.filler_stage = 0
        self.honed_reavers = 0.0
        self.honed_steel = 0.0
        self.hunters_instinct = 0.0
        self.hunters_instinct_applied = False
        self.swiftscaled = 0.0
        # rotation stuff buffs
        self.hindstung_venom = 0.0
        self.hindsbane_venom = 0.0
        self.flanksbane_venom = 0.0
        self.flankstung_venom = 0.0
        self.death_rattle_ready = 0


    def get_max_actions(self):
        return len(self.actions)
    
    def get_state_shape(self):
        state = self.state()
        return state.shape[0]
    
    # Will return an array filled with truth and false for each action at that moment
    # Will also return an array of valid action ids
    def valid_actions(self):
        action_mask = []
        actions_allowed = []
        for a_id in range(len(self.actions)):
            a_name = self.actions[a_id][0]
            if self.filler_stage == 0 and (a_name == 'steel_fangs' or a_name == 'reaving_fangs'):
                action_mask.append(True)
                actions_allowed.append(a_id)
            elif self.filler_stage == 1 and (a_name == 'hunters_sting' or a_name == 'swiftskins_sting'):
                action_mask.append(True)
                actions_allowed.append(a_id)
            elif self.filler_stage == 2 and (a_name == 'flanksting_strike' or a_name == 'flanksbane_fang'):
                action_mask.append(True)
                actions_allowed.append(a_id)
            elif self.filler_stage == 3 and (a_name == 'hindsting_strike' or a_name == 'hindsbane_fang'):
                action_mask.append(True)
                actions_allowed.append(a_id)
            elif self.death_rattle_ready:
                action_mask.append(True)
                actions_allowed.append(a_id)
            else:
                action_mask.append(False)
        return action_mask, actions_allowed

    # Takes an int representing an action to take
    # Returns a tuple of (reward-time cost, reward, rolled damage)
    def step(self, action: int, _verbose = False):
        if action < 0 or action > self.get_max_actions()-1:
            print(f'Invalid action {action} chosen')
            return -1.0, 0.0, 0.0
        action_name = self.actions[action][0]
        action_reward = 0.0
        time_malus = 0.0
        action_success = False

        # check here and consume relevant ogcd buffs if they are not used immediately
        if not action_name == 'death_rattle' and self.death_rattle_ready == 1:
            self.death_rattle_ready = 0

        # All buffs given from actions should be applied here, consume or reset depending on interactions
        #  with the rest of the toolkit
        if action_name == 'steel_fangs':
            if self.filler_stage == 0:
                # step time forward to the next gcd spot, tick buffs as needed
                # Adjusts gcd lock for this action if given
                time_malus = self.valid_action(2.5)
                bonus = 0
                if self.honed_steel > 0:
                    self.honed_steel = 0
                    bonus = 100
                # we apply buffs here to prevent premature ticking on them
                self.honed_reavers = 60.0
                self.filler_stage = 1

                # Time step gets called later.
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

                action_success = True
                action_reward = 200 + bonus
        elif action_name == 'hunters_sting':
            if self.filler_stage == 1:
                time_malus = self.valid_action()
                if self.hunters_instinct <= 0.0:
                    self.hunters_instinct_applied = True
                self.hunters_instinct = 40.0
                self.filler_stage = 2

                action_success = True
                action_reward = 300
        elif action_name == 'swiftskins_sting':
            if self.filler_stage == 1:
                time_malus = self.valid_action()
                self.swiftscaled = 40.0
                self.filler_stage = 3

                action_success = True
                action_reward = 300
        # Flanks
        elif action_name == 'flanksting_strike':
            if self.filler_stage == 2:
                time_malus = self.valid_action()
                bonus = 0
                if self.flankstung_venom > 0.0:
                    bonus = 100
                    self.flankstung_venom = 0.0
                self.hindstung_venom = 60.0
                self.flanksbane_venom = 0.0
                self.hindsbane_venom = 0.0
                self.filler_stage = 0
                self.death_rattle_ready = 1

                action_success = True
                action_reward = 400 + bonus
        elif action_name == 'flanksbane_fang':
            if self.filler_stage == 2:
                time_malus = self.valid_action()
                bonus = 0
                if self.flanksbane_venom > 0.0:
                    bonus = 100
                    self.flanksbane_venom = 0.0
                self.hindsbane_venom = 60.0
                self.hindstung_venom = 0.0
                self.flankstung_venom = 0.0
                self.filler_stage = 0
                self.death_rattle_ready = 1

                action_success = True
                action_reward = 400 + bonus
        # Rears
        elif action_name == 'hindsting_strike':
            if self.filler_stage == 3:
                time_malus = self.valid_action()
                bonus = 0
                if self.hindstung_venom > 0.0:
                    bonus = 100
                    self.hindstung_venom = 0.0
                self.flanksbane_venom = 60.0
                self.hindsbane_venom = 0.0
                self.flankstung_venom = 0.0
                self.filler_stage = 0
                self.death_rattle_ready = 1

                action_success = True
                action_reward = 400 + bonus
        elif action_name == 'hindsbane_fang':
            if self.filler_stage == 3:
                time_malus = self.valid_action()
                bonus = 0
                if self.hindsbane_venom > 0.0:
                    bonus = 100
                    self.hindsbane_venom = 0.0
                self.flankstung_venom = 60.0
                self.hindstung_venom = 0.0
                self.flanksbane_venom = 0.0
                self.filler_stage = 0
                self.death_rattle_ready = 1

                action_success = True
                action_reward = 400 + bonus
        #OGCDS
        elif action_name == 'death_rattle':
            if self.death_rattle_ready == 1:
                self.death_rattle_ready = 0

                action_success = True
                action_reward = 280
        
        # This is to handle the initial application of hunter's sting.
        # Which shouldn't affect the first instance of damage applied.
        if self.hunters_instinct > 0:
            if not self.hunters_instinct_applied:
                action_reward = action_reward * 1.1
            self.hunters_instinct_applied = False
        
        if _verbose:
            print(f'Took action: {action}-{action_name} @ {self.time:.3f}')

        # on fail / bad action, step forward 100 ms, applies a negative reward as well
        if not action_success:
            time_malus += self.invalid_action() + 9.0
        else:
            # otherwise time step to the next free animation slot, tick buffs as needed
            time_malus += self.action_lock(self.action_lock_duration)

        damage = self.compute_damage(action_reward)
        reward = action_reward if action_reward < 0 else action_reward / 10.0
        reward = reward - time_malus
        reward = reward / 5.0
        #print(action_reward, time_malus)

        return reward, action_reward, damage
    
    # Hard lock for 400 ms to punish incorrect flow.
    def invalid_action(self):
        delta_time = 0.400
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
        time_malus = 0
        # Roll the gcd forward to the next possible time
        delta_time = max(self.gcd - self.gcd_roll, 0) # accounts for clipping
        time_malus = self.time_step(delta_time)
        self.gcd_roll = 0.0 # Then set to zero

        # if we have haste buff or a longer gcd then adjust gcd lock here at the end
        self.gcd = self.compute_gcd(gcd_lock, self.sks, 15 if self.swiftscaled > 0 else 0)
        return time_malus  
    
    # handle all animation action locks
    def action_lock(self, lock_time):
        # roll gcd for action lock time, clip gcd if necessary
        delta_time = lock_time
        return self.time_step(delta_time)
    
    # subtract 10 potency per 100 ms of action
    def time_step(self, delta_time):
        self.gcd_roll += delta_time
        self.time += delta_time
        # TODO handle all buff timers and update here
        self.honed_reavers = max(0, self.honed_reavers - delta_time)
        self.honed_steel = max(0, self.honed_steel - delta_time)
        self.hunters_instinct = max(0, self.hunters_instinct - delta_time)
        self.swiftscaled = max(0, self.swiftscaled - delta_time)
        self.flanksbane_venom = max(0, self.flanksbane_venom - delta_time)
        self.flankstung_venom = max(0, self.flankstung_venom - delta_time)
        self.hindsbane_venom = max(0, self.hindsbane_venom - delta_time)
        self.hindstung_venom = max(0, self.hindstung_venom - delta_time)

        return delta_time / 0.100 # every 100 ms incurs 1 potency cost to punish hard clipping.
    
    def compute_damage(self, potency):
        variance = np.random.uniform(0.95, 1.05)
        damage = np.floor(potency * variance * 100) / 100
        return damage
    
    def state(self):
        # We should return the state as a tensor for ease of use
        _state = [
            #self.time,
            self.gcd,
            self.gcd_roll,
            #self.filler_stage,
            self.honed_reavers,
            self.honed_steel,
            self.hunters_instinct,
            self.swiftscaled,
            self.flanksbane_venom,
            self.flankstung_venom,
            self.hindsbane_venom,
            self.hindstung_venom,
            self.death_rattle_ready
        ]
        _filler_stage = self.one_hot_encode(self.filler_stage, 4)
        _state += _filler_stage
        return torch.tensor(_state, dtype=torch.float32, device=torch.device('cpu'))
    
    def is_done(self):
        return False