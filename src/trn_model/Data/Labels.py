class Labels():

    classes = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3,
                   'CricketBowling': 4, 'CricketShot': 5, 'CliffDiving': 6, 'Diving': 7,
                   'FrisbeeCatch': 8, 'GolfSwing': 9, 'HammerThrow': 10, 'HighJump': 11,
                   'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'SoccerPenalty': 15,
                   'Shotput': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}

    @staticmethod
    def get_classes():
        return Labels.classes
