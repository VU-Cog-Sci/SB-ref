from exptools.core.session import EyelinkSession
from trial import QNATrial
from psychopy import clock, sound
from psychopy.visual import ImageStim, MovieStim
import numpy as np
import os
import exptools
import json
import glob
import scipy.io as io
from psychopy import logging, visual, event


class QNASession(EyelinkSession):

    def __init__(self, *args, **kwargs):

        super(QNASession, self).__init__(*args, **kwargs)

        for argument in ['size_fixation_deg', 'language']:
            value = kwargs.pop(argument, self.config.get('stimuli', argument))
            setattr(self, argument, value)
        for argument in ['fixation_time', 'stimulus_time']:
            value = kwargs.pop(argument, self.config.get('timing', argument))
            setattr(self, argument, value)

        self.fixation = visual.GratingStim(self.screen,
                                           tex='sin',
                                           mask='raisedCos',
                                           size=self.deg2pix(self.size_fixation_deg),
                                           texRes=512,
                                           color=self.foreground_color,
                                           sf=0,
                                           maskParams={'fringeWidth': 0.4})

        if self.index_number != 0:
            instruction_text = 'Please follow the spoken instructions,\n and fixate during the entire scan. \nWaiting for Scanner'
        else:
            instruction_text = 'Please fixate during the entire scan. \nWaiting for Scanner'

        self.instruction = visual.TextStim(self.screen, 
            text = instruction_text, 
            font = 'Helvetica Neue',
            pos = (0, self.deg2pix(2)),
            italic = True, 
            height = 20, 
            alignHoriz = 'center',
            color=(-1,-1,-1))
        self.instruction.setSize((1200,50))

        self.create_trials()

        self.stopped = False

    def create_trials(self):
        """creates trials by loading a list of jpg files from the img/ folder"""

        if self.index_number == 1:
            self.sound_indices = [1,2,3,4]
        elif self.index_number == 2:
            self.sound_indices = [5,6,7,9]
        elif self.index_number == 3:
            self.sound_indices = [1,2,3,4]
        elif self.index_number == 4:
            self.sound_indices = [5,6,7,9]

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
            elif ti == len(self.sound_stims)-1:
                phase_durations = [-0.001, self.fixation_time, self.stimulus_time, self.fixation_time*3]
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
