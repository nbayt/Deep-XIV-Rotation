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
        self.gcd_roll = 0.0
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

    def step(self, action: int):
        if action < 0 or action > self.get_max_actions()-1:
            print(f'Invalid action {action} chosen')
            return
        action_reward = 0.0
        time_malus = 0.0
        action_time = 0.0
        action_success = True
        # All buffs given from actions should be applied here, consume or reset depending on interactions
        #  with the rest of the toolkit
        if action[0] == 'steel_fangs':
            if self.filler_stage == 0:
                bonus = 0
                if self.honed_steel > 0:
                    self.honed_steel = 0
                    bonus = 100
                self.honed_reavers = 60.0
                time_malus = self.valid_action(self.action_lock_duration)
                self.filler_stage = 1
            else:
                pass
        # on fail / bad action, step forward 100 ms
        if not action_success:
            time_malus = self.invalid_action()

    def invalid_action(self):
        # Hard lock for 100 ms to punish incorrect flow.
        delta_time = 100
        return self.time_step(delta_time)

    def valid_action(self, lock_time, is_ogcd = False):
        delta_time = 0
        if not is_ogcd:
            # Roll the gcd forward to the next possible time
            delta_time = max(self.gcd - self.gcd_roll, 0) # accounts for clipping
        else:
            # roll gcd for action lock time, clip gcd if necessary
            delta_time = lock_time
        return self.time_step(delta_time)
    
    def compute_damage(self, potency):
        pass

    # subtract 10 potency per 100 ms of action
    def time_step(self, delta_time):
        self.gcd_roll += delta_time
        self.time += delta_time
        # TODO handle all buff timers and update here

        return delta_time / 100 # every 100 ms incurs 1 potency cost to punish hard clipping.

    def state(self):
        pass