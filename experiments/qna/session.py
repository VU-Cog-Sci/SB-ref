from exptools.core.session import EyelinkSession
from trial import QNATrial
from psychopy import clock, sound
from psychopy.visual import ImageStim, MovieStim
import numpy as np
import os
import exptools
import json
import glob


class QNASession(EyelinkSession):

    def __init__(self, *args, **kwargs):

        super(QNASession, self).__init__(*args, **kwargs)

        for argument in ['size_fixation_deg', 'language']:
            value = kwargs.pop(argument, self.config.get('stimuli', argument))
            setattr(self, argument, value)
        for argument in ['fixation_time', 'stimulus_time']:
            value = kwargs.pop(argument, self.config.get('timing', argument))
            setattr(self, argument, value)
        self.create_trials()

        self.stopped = False

    def create_trials(self):
        """creates trials by loading a list of jpg files from the img/ folder"""

        if self.index_number == 1:
            self.sound_indices = [1,2,3,4]
        elif self.index_number == 2:
            self.sound_indices = [5,6,7,8]
        elif self.index_number == 3:
            self.sound_indices = [1,2,3,4]
        elif self.index_number == 4:
            self.sound_indices = [5,6,7,8]


        sound_files = [os.path.join(os.path.abspath(os.getcwd()), 'sounds', 'wav', 'Q%i_english.wav'%s) for s in self.sound_indices]

        self.sound_stims = [sound.Sound(isf) for isf in sound_files]
        self.trial_order = np.arange(len(self.sound_stims))


    def run(self):
        """docstring for fname"""
        # cycle through trials

        for ti in np.arange(len(self.sound_stims)):

            parameters = {'stimulus': self.trial_order[ti], 'sound':self.trial_order[ti]}

            # parameters.update(self.config)
            if ti == 0:
                phase_durations = [1800, self.fixation_time, self.stimulus_time, self.fixation_time]
            else:
                phase_durations = [-0.001, self.fixation_time, self.stimulus_time, self.fixation_time]

            trial = QNATrial(phase_durations=phase_durations,
                           screen=self.screen,
                           session=self,
                           parameters=parameters,
                           tracker=self.tracker)
            trial.run(ID=ti)

            if self.stopped == True:
                break

        self.close()
