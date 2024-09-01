class Viper:
    def __init__(self):
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

        self.gcd = 2.50
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
        action_time = 0.0
        action_success = True
        if action[0] == 'steel_fangs':
            if self.filler_stage == 0:
                pass
            else:
                pass
        # on fail / bad action, step forward 100 ms
        if not action_success:
            pass

    def invalid_action(self):
        pass

    def valid_action(self, is_ogcd = False):
        pass

    # subtract 10 potency per 100 ms of action
    def time_step(self, time):
        pass

    def state(self):
        pass