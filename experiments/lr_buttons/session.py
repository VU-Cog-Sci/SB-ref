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

        nr_trials = self.n_trials * 4   # nr of trials is four times the yml definition
        all_trial_itis = np.r_[np.ones(nr_trials/2)*0, np.ones(nr_trials/4)*0.5, np.ones(nr_trials/8), np.ones(nr_trials/16)*2, np.ones(nr_trials/16)*4]
        all_trial_itis = all_trial_itis / all_trial_itis.mean() 
        all_trial_itis *= self.inter_trial_interval_mean
        all_trial_itis += self.inter_trial_interval_min 
        np.random.shuffle(all_trial_itis)

        sides = [-1,1]
        shapes = [0,1]
        self.trial_parameters = []
        for i in xrange(self.n_trials):
            for j, side in enumerate(sides):
                for k, shape in enumerate(shapes):
                    this_trial_dict = {
                        'shape': shape,
                        'side': side,
                        'iti' : all_trial_itis[len(self.trial_parameters)],  
                        'finger_instruction': self.index_number
                        }
                    this_trial_dict.update(self.config_parameters)
                    self.trial_parameters.append(this_trial_dict)

        random.shuffle(self.trial_parameters)

        self.trial_parameters[0]['fixation_duration'] = self.intro_extro_duration
        for tp in self.trial_parameters:
            tp['wait_duration'] = -0.001
        self.trial_parameters[0]['wait_duration'] = 1200
        self.trial_parameters[-1]['iti'] = self.intro_extro_duration + self.trial_parameters[-1]['iti']

    def setup_stimuli(self):
        size_fixation_pix = self.deg2pix(self.size_fixation_deg)
        self.fixation = visual.GratingStim(self.screen,
                                           tex='sin',
                                           mask='raisedCos',
                                           size=size_fixation_pix,
                                           texRes=512,
                                           color='white',
                                           sf=0,
                                           maskParams={'fringeWidth': 0.4})

        self.square_stim = visual.Rect(self.screen, 
                                width=self.deg2pix(self.shape_size),
                                height=self.deg2pix(self.shape_size),
                                lineWidth=self.deg2pix(self.shape_lw), 
                                lineColor='white',
                                fillColor='black')

        self.circle_stim = visual.Circle(self.screen, 
                                radius=self.deg2pix(self.shape_size)/2.0,
                                lineWidth=self.deg2pix(self.shape_lw), 
                                lineColor='white',
                                fillColor='black')

        self.shape_stims = [self.square_stim, self.circle_stim]

        if self.language == 'EN':
            this_instruction_string = """When you see a {target}, press the button with your {insert_string} index finger 
If you see a {distractor}, press nothing. 
Waiting for the scanner to start."""
            if self.index_number < 3:
                insert_string = 'LEFT'
            elif self.index_number >= 3:
                insert_string = 'RIGHT'
            if self.index_number % 2 == 0:
                target = 'SQUARE'
                distractor = 'CIRCLE'
            else:
                target = 'CIRCLE'
                distractor = 'SQUARE'           
        elif self.language == 'IT':
            this_instruction_string = """Quando vedi un {target}quadrato, premi il pulsante con il dito indice %s
Se vedi un {distractor}cerchio, non premere nulla.
In attesa che lo scanner inizi."""
            if self.index_number < 3:
                insert_string = 'SINISTRO'
            elif self.index_number >= 3:
                insert_string = 'DESTRO'
            if self.index_number % 2 == 0:
                target = 'QUADRATO'
                distractor = 'CERCHIO'
            else:
                target = 'CERCHIO'
                distractor = 'QUADRATO'    

        self.instruction = visual.TextStim(self.screen, 
            text = this_instruction_string.format(insert_string=insert_string, target=target, distractor=distractor), 
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
