from exptools.core.session import EyelinkSession
from trial import LR_IMTrial
from psychopy import visual, clock
import numpy as np
import os
import exptools
import json
import glob
import random


class LR_IMSession(EyelinkSession):

    def __init__(self, *args, **kwargs):

        super(LR_IMSession, self).__init__(*args, **kwargs)

        self.config_parameters = {}
        for argument in ['size_fixation_deg', 'shape_ecc', 'shape_size',
                     'shape_lw', 'n_trials', 'language']:
            value = kwargs.pop(argument, self.config.get('stimuli', argument))
            setattr(self, argument, value)
            self.config_parameters.update({argument: value})

        for argument in ['fixation_duration', 'inter_trial_interval_mean', 'inter_trial_interval_min',
                     'intro_extro_duration', 'stimulus_duration']:
            value = kwargs.pop(argument, self.config.get('timing', argument))
            setattr(self, argument, value)
            self.config_parameters.update({argument: value})

        self.create_trials()
        self.setup_stimuli()

        self.stopped = False


    def create_trials(self):
        """creates trials by creating a restricted random walk through the display from trial to trial"""

        sides = [-1,1]
        shapes = [0,1]
        self.trial_parameters = []
        for i in xrange(self.n_trials):
            for j, side in enumerate(sides):
                for k, shape in enumerate(shapes):
                    this_trial_dict = {
                        'shape': shape,
                        'side': side,
                        'iti' : self.inter_trial_interval_min + \
                        np.random.exponential(self.inter_trial_interval_mean),  
                        'finger_instruction': self.index_number
                        }
                    this_trial_dict.update(self.config_parameters)
                    self.trial_parameters.append(this_trial_dict)

        random.shuffle(self.trial_parameters)

        self.trial_parameters[0]['fixation_duration'] = self.intro_extro_duration
        for tp in self.trial_parameters:
            tp['wait_duration'] = -0.001
        self.trial_parameters[0]['wait_duration'] = 1200
        self.trial_parameters[-1]['iti'] = self.intro_extro_duration

    def setup_stimuli(self):
        size_fixation_pix = self.deg2pix(self.size_fixation_deg)

        self.fixation = visual.GratingStim(self.screen,
                                           tex='sin',
                                           mask='circle',
                                           size=size_fixation_pix,
                                           texRes=512,
                                           color='white',
                                           sf=0)

        self.square_stim = visual.Rect(self.screen, 
                                width=self.deg2pix(self.shape_size),
                                height=self.deg2pix(self.shape_size),
                                lineWidth=self.deg2pix(self.shape_lw), 
                                lineColor='white',
                                fillColor='red')

        self.circle_stim = visual.Circle(self.screen, 
                                radius=self.deg2pix(self.shape_size)/2.0,
                                lineWidth=self.deg2pix(self.shape_lw), 
                                lineColor='white',
                                fillColor='green')

        self.shape_stims = [self.square_stim, self.circle_stim]

        if self.language == 'EN':
            this_instruction_string = """When you see a square, press the button with your %s index finger 
If you see a circle, press nothing. 
Waiting for the scanner to start."""
            if self.index_number == -1:
                insert_string = 'LEFT'
            elif self.index_number == 1:
                insert_string = 'RIGHT'
        elif self.language == 'IT':
            this_instruction_string = """Quando vedi un quadrato, premi il pulsante con il dito indice %s
Se vedi un cerchio, non premere nulla.
In attesa che lo scanner inizi."""
            if self.index_number == -1:
                insert_string = 'SINISTRO'
            elif self.index_number == 1:
                insert_string = 'DESTRO'
        self.instruction = visual.TextStim(self.screen, 
            text = this_instruction_string%insert_string, 
            font = 'Helvetica Neue',
            pos = (0, 0),
            italic = True, 
            height = 20, 
            alignHoriz = 'center',
            color=(1,0,0))
        self.instruction.setSize((1200,50))

    def run(self):
        """run the session"""
        # cycle through trials


        for trial_id, parameters in enumerate(self.trial_parameters):

            trial = LR_IMTrial(ti=trial_id,
                           config=self.config,
                           screen=self.screen,
                           session=self,
                           parameters=parameters,
                           tracker=self.tracker)
            trial.run()

            if self.stopped == True:
                break

        self.close()
